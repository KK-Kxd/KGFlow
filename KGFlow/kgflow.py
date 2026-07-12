from kg_explore import process_query
from kg_alignment import extract_entities_from_chains, align_entity_graphs_pairwise, has_extractable_chains
from KGs.hetionet import HetionetDatabase
from KGs.umls import UMLSDatabase
from KGs.primeKG import PrimeDatabase
from local_llm import ChatModel
from kg_fuser import GraphFuser
from Dataset.QADataset import QADataset
from Dataset.PubMedQADataset import PubMedQADataset
import promptTemplate
from tqdm import tqdm
import concurrent.futures
import argparse
import copy
from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
import torch
import uuid

from typing import List, Tuple, Any

torch.manual_seed(42)

DEFAULT_MODEL = "llama3.1-8b"

def parse_answer_choice(response: str) -> str:
        match = re.search(r"Answer:\s*([A-D])", response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        match = re.search(r"\b([A-D])\b", response.strip(), re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return "Parsing Failed"

def predict_final_answer(query_with_options: str, relevant_paths_with_descriptions: List[Tuple[List[Tuple], str]], model: Any) -> Tuple[str, str]:
        reasoning_context = []
        for i, (path, description) in enumerate(relevant_paths_with_descriptions):
            context_item = f"Evidence Path {i+1}:\n- Reasoning Chain: {path}\n- Explanation: {description}"
            reasoning_context.append(context_item)
        
        formatted_context = "\n\n".join(reasoning_context)

        prompt_template = """
        You are an expert in biomedical question answering.
        Given the following inputs:
        A medical query with multiple-choice options (A, B, C, D).
        A set of relevant reasoning paths retrieved from a biomedical knowledge graph. Each reasoning path has a corresponding natural language explanation.
        
        Based solely on the evidence provided in the reasoning paths, choose the most appropriate answer (A, B, C, or D).
        Do not guess beyond the evidence. If multiple options are mentioned, choose the one best supported by the reasoning paths.
        Please provide the answer in the following format: "Answer: A/B/C/D".

        Q: {}
        
        Reasoning Paths:
        {}
        """
        prompt = prompt_template.format(query_with_options, formatted_context)

        response = model.generate_response(prompt, 0.0)
        return parse_answer_choice(response), response

def predict_direct_answer(query_with_options: str, model: Any) -> Tuple[str, str]:
        prompt = promptTemplate.answer_without_prompt.format(query_with_options)
        response = model.generate_response(prompt, 0.0)
        return parse_answer_choice(response), response

def process_and_extract_worker(query, args, db_instance, model_instance, sample_no=None, kg_name=None):
    kg_label = kg_name or db_instance.get_name()
    prefix = f"[sample {sample_no}][{kg_label}]"
    worker_args = copy.copy(args)
    kg_worker_timeout = getattr(worker_args, "kg_worker_timeout", None)
    if kg_worker_timeout and kg_worker_timeout > 0:
        worker_args._kgflow_deadline = time.monotonic() + kg_worker_timeout

    started_at = time.monotonic()
    print(f"{prefix} process_query start", flush=True)
    success, topic_ent, chains = process_query(query, worker_args, db_instance, model_instance)
    print(
        f"{prefix} process_query done success={success} topic_entities={len(topic_ent)} hops={len(chains)} "
        f"elapsed={time.monotonic() - started_at:.1f}s",
        flush=True,
    )

    e_l = []
    if has_extractable_chains(chains):
        extract_started_at = time.monotonic()
        e_l = extract_entities_from_chains(
            chains,
            model_instance,
            db_instance,
            max_entities=worker_args.max_alignment_entities,
            description_workers=worker_args.entity_description_workers,
            describe_entities=not getattr(worker_args, "skip_entity_descriptions", False),
        )
        print(
            f"{prefix} entity extraction done entities={len(e_l)} "
            f"elapsed={time.monotonic() - extract_started_at:.1f}s",
            flush=True,
        )
    return success, topic_ent, chains, e_l

def make_jsonable(value):
    if isinstance(value, BaseException):
        return {
            "type": type(value).__name__,
            "message": str(value),
        }
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(key): make_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(item) for item in value]
    if isinstance(value, set):
        return [make_jsonable(item) for item in sorted(value, key=str)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_jsonable(data), f, ensure_ascii=False, indent=2)

def append_jsonl(path, data):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(make_jsonable(data), ensure_ascii=False) + "\n")

class StageWriter:
    def __init__(self, stage_paths):
        self.stage_paths = stage_paths
        self._locks = {
            stage_name: threading.Lock()
            for stage_name in ("knowledge_explorer", "graph_aligner", "contextual_pruner", "answers")
        }

    def append(self, stage_name, record):
        path = self.stage_paths[stage_name]
        with self._locks[stage_name]:
            append_jsonl(path, record)

def record_stage(records, stage_name, record, stage_writer=None):
    records[stage_name].append(record)
    if stage_writer is not None:
        stage_writer.append(stage_name, record)

def select_indices(data_len, sample_size, sample_seed):
    indices = list(range(data_len))
    if sample_size is None or sample_size <= 0 or sample_size >= data_len:
        return indices
    return random.Random(sample_seed).sample(indices, sample_size)

def load_sample_indices(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("indices", [])
    return [int(index) for index in data]

def normalize_topic_entity(entity):
    if isinstance(entity, dict):
        source = entity.get("source") or entity.get("kg") or entity.get("database") or "unknown"
        name = entity.get("name") or entity.get("entity") or entity.get("ent_id") or entity.get("id")
        return (str(source), str(name)) if name else None
    if isinstance(entity, (list, tuple)) and len(entity) >= 2:
        return (str(entity[0]), str(entity[1]))
    if entity:
        return ("unknown", str(entity))
    return None

def build_chat_model(args, device):
    return ChatModel(args.model, args.model, max_token=args.max_token, device=device)

def resolve_worker_cuda_visible_device(device: str) -> str:
    match = re.fullmatch(r"cuda:(\d+)", str(device).strip())
    if not match:
        raise ValueError(
            f"Local KG worker device must be an explicit cuda:N value, got {device!r}. "
            "Use --umls_device/--primekg_device/--hetionet_device to pin each worker."
        )

    logical_index = int(match.group(1))
    parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if parent_visible:
        visible_entries = [entry.strip() for entry in parent_visible.split(",") if entry.strip()]
        if visible_entries and visible_entries[0] != "-1":
            if logical_index >= len(visible_entries):
                raise ValueError(
                    f"{device} is outside parent CUDA_VISIBLE_DEVICES={parent_visible!r}"
                )
            return visible_entries[logical_index]

    return str(logical_index)

class KGSubprocessClient:
    def __init__(self, kg_key, kg_name, config_path, log_path, device, args):
        self.kg_key = kg_key
        self.kg_name = kg_name
        self.config_path = config_path
        self.log_path = log_path
        self.device = device
        self._lock = threading.Lock()
        self._stderr = open(log_path, "a", encoding="utf-8")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = resolve_worker_cuda_visible_device(device)

        worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kg_worker_process.py")
        self.process = subprocess.Popen(
            [sys.executable, "-u", worker_script, "--config", config_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=env,
        )

    @property
    def pid(self):
        return self.process.pid

    def _request(self, request_type, **payload):
        request_id = str(uuid.uuid4())
        request = {"id": request_id, "type": request_type}
        request.update(payload)

        with self._lock:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"{self.kg_name} worker exited with code {self.process.returncode}; see {self.log_path}"
                )

            try:
                self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                self.process.stdin.flush()
            except BrokenPipeError as exc:
                raise RuntimeError(
                    f"{self.kg_name} worker pipe closed; see {self.log_path}"
                ) from exc

            while True:
                line = self.process.stdout.readline()
                if not line:
                    exit_code = self.process.poll()
                    raise RuntimeError(
                        f"{self.kg_name} worker stopped before responding "
                        f"(exit_code={exit_code}); see {self.log_path}"
                    )
                try:
                    response = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if response.get("id") != request_id:
                    raise RuntimeError(
                        f"{self.kg_name} worker returned mismatched response id; see {self.log_path}"
                    )
                if response.get("ok"):
                    return response.get("result")
                error = response.get("error") or {}
                error_type = error.get("type", "WorkerError")
                message = error.get("message", "unknown worker error")
                raise RuntimeError(f"{self.kg_name} worker {error_type}: {message}")

    def process_query(self, query, sample_no=None, dataset_index=None):
        return self._request(
            "process_query",
            query=query,
            sample_no=sample_no,
            dataset_index=dataset_index,
        )

    def extract_entities(self, chains, sample_no=None, dataset_index=None):
        return self._request(
            "extract_entities",
            chains=chains,
            sample_no=sample_no,
            dataset_index=dataset_index,
        )

    def generate(self, prompt, temperature=1.0):
        return self._request("generate", prompt=prompt, temperature=temperature)

    def close(self):
        try:
            if self.process.poll() is None:
                try:
                    self._request("shutdown")
                except Exception:
                    pass
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=10)
        finally:
            for stream in (self.process.stdin, self.process.stdout):
                try:
                    if stream:
                        stream.close()
                except Exception:
                    pass
            try:
                self._stderr.close()
            except Exception:
                pass

class KGPooledClient:
    def __init__(self, clients):
        if not clients:
            raise ValueError("KGPooledClient requires at least one worker")
        self.clients = clients
        self._available = queue.Queue()
        for client in clients:
            self._available.put(client)

    @property
    def pids(self):
        return [client.pid for client in self.clients]

    def _with_client(self, method_name, *args, **kwargs):
        client = self._available.get()
        try:
            return getattr(client, method_name)(*args, **kwargs)
        finally:
            self._available.put(client)

    def process_query(self, query, sample_no=None, dataset_index=None):
        return self._with_client("process_query", query, sample_no, dataset_index)

    def extract_entities(self, chains, sample_no=None, dataset_index=None):
        return self._with_client("extract_entities", chains, sample_no, dataset_index)

    def generate(self, prompt, temperature=1.0):
        return self._with_client("generate", prompt, temperature)

    def close(self):
        for client in self.clients:
            client.close()

class WorkerModelProxy:
    def __init__(self, worker_client):
        self.worker_client = worker_client

    def generate_response(self, input_text: str, temperature: float = 1.0) -> str:
        return self.worker_client.generate(input_text, temperature)

def start_kg_subprocess_workers(args, output_root):
    workers_dir = os.path.join(output_root, "workers")
    os.makedirs(workers_dir, exist_ok=True)

    worker_specs = [
        ("umls", "umls", args.umls_device, args.umls_workers),
        ("primekg", "primeKG", args.primekg_device, args.primekg_workers),
        ("hetionet", "hetionet", args.hetionet_device, args.hetionet_workers),
    ]
    workers = {}
    worker_info = {}
    base_args = vars(args).copy()

    for kg_key, kg_name, device, worker_count in worker_specs:
        clients = []
        worker_entries = []
        for worker_index in range(worker_count):
            suffix = "" if worker_count == 1 else f"_{worker_index}"
            config_path = os.path.join(workers_dir, f"{kg_key}{suffix}_config.json")
            log_path = os.path.join(workers_dir, f"{kg_key}{suffix}.log")
            config = {
                "kg_key": kg_key,
                "kg_name": kg_name,
                "worker_index": worker_index,
                "device": device,
                "args": base_args,
            }
            write_json(config_path, config)
            client = KGSubprocessClient(kg_key, kg_name, config_path, log_path, device, args)
            clients.append(client)
            worker_entry = {
                "index": worker_index,
                "pid": client.pid,
                "device": device,
                "config": config_path,
                "log": log_path,
                "cuda_visible_devices": resolve_worker_cuda_visible_device(device),
            }
            worker_entries.append(worker_entry)
            print(
                f"[worker {kg_key}:{worker_index}] pid={client.pid} device={device} log={log_path}",
                flush=True,
            )

        workers[kg_key] = KGPooledClient(clients)
        worker_info[kg_key] = {
            "name": kg_name,
            "count": worker_count,
            "pids": [entry["pid"] for entry in worker_entries],
            "device": device,
            "workers": worker_entries,
            "pid": worker_entries[0]["pid"],
            "config": worker_entries[0]["config"],
            "log": worker_entries[0]["log"],
            "cuda_visible_devices": resolve_worker_cuda_visible_device(device),
        }

    write_json(os.path.join(workers_dir, "worker_info.json"), worker_info)
    return workers, worker_info

def close_kg_subprocess_workers(workers):
    for worker in workers.values():
        worker.close()

def process_sample(sample_no, idx, dataset, data, args, tasks_to_run, kg_names, model_umls, run_id, stage_writer=None):
    print(f"[sample {sample_no}] start dataset_index={idx}", flush=True)
    records = {
        "knowledge_explorer": [],
        "graph_aligner": [],
        "contextual_pruner": [],
        "answers": [],
    }

    d = data[idx]
    if 'text' in d and 'answer' in d:
        query = d['text']
        label = d['answer']
    else:
        answer_record = {
            "run_id": run_id,
            "dataset": dataset,
            "sample_no": sample_no,
            "dataset_index": idx,
            "error": "Missing text or answer field",
        }
        record_stage(records, "answers", answer_record, stage_writer)
        return {
            "sample_no": sample_no,
            "dataset_index": idx,
            "records": records,
            "completed": False,
            "skipped": True,
            "is_correct": False,
            "final_answer": None,
        }

    all_results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks_to_run)) as executor:
        future_to_name = {
            executor.submit(client.process_query, query, sample_no, idx): name
            for name, client in tasks_to_run
        }

        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                result = future.result()
                result.setdefault("error", None)
                all_results[name] = result
            except Exception as exc:
                all_results[name] = {"error": exc}
                print(f"[sample {sample_no}][{name}] failed: {type(exc).__name__}: {exc}", flush=True)
    print(f"[sample {sample_no}] knowledge_explorer done", flush=True)

    knowledge_record = {
        "run_id": run_id,
        "dataset": dataset,
        "sample_no": sample_no,
        "dataset_index": idx,
        "query": query,
        "label": label,
        "results": all_results,
    }
    record_stage(records, "knowledge_explorer", knowledge_record, stage_writer)

    umls_result = all_results.get(kg_names["umls"])
    primekg_result = all_results.get(kg_names["primekg"])
    hetionet_result = all_results.get(kg_names["hetionet"])

    topic_ent = []
    for result in (umls_result, primekg_result, hetionet_result):
        if result and result.get("topic_ent"):
            for entity in result.get("topic_ent"):
                normalized_entity = normalize_topic_entity(entity)
                if normalized_entity:
                    topic_ent.append(normalized_entity)
    topic_ent = list(dict.fromkeys(topic_ent))

    # Only use available results
    umls_entities = umls_result.get("entities") if umls_result else None
    primekg_entities = primekg_result.get("entities") if primekg_result else None
    hetionet_entities = hetionet_result.get("entities") if hetionet_result else None

    entity_counts = {
        kg_names["umls"]: len(umls_entities or []),
        kg_names["primekg"]: len(primekg_entities or []),
        kg_names["hetionet"]: len(hetionet_entities or []),
    }
    entity_inputs = {
        kg_names["umls"]: umls_entities or [],
        kg_names["primekg"]: primekg_entities or [],
        kg_names["hetionet"]: hetionet_entities or [],
    }
    kgs_with_entities = [name for name, entities in entity_inputs.items() if entities]

    if len(kgs_with_entities) < 2:
        fallback_reason = "Fewer than two KGs returned entities"
        print(f"Warning: {fallback_reason} for query: {query[:50]}... using direct LLM answer")

        graph_record = {
            "run_id": run_id,
            "dataset": dataset,
            "sample_no": sample_no,
            "dataset_index": idx,
            "aligned_triplets": [],
            "entity_counts": entity_counts,
            "kgs_with_entities": kgs_with_entities,
            "skipped": True,
            "reason": fallback_reason,
        }
        record_stage(records, "graph_aligner", graph_record, stage_writer)
        context_record = {
            "run_id": run_id,
            "dataset": dataset,
            "sample_no": sample_no,
            "dataset_index": idx,
            "topic_ent": topic_ent,
            "candidate_paths": [],
            "paths_with_desc": [],
            "relevant_paths": [],
            "skipped": True,
            "reason": fallback_reason,
        }
        record_stage(records, "contextual_pruner", context_record, stage_writer)

        final_answer, final_response = predict_direct_answer(query, model_umls)
        predict_answer = final_answer if final_answer in {"A", "B", "C", "D"} else "None"
        is_correct = predict_answer == label
        print(f"[sample {sample_no}] direct answer done", flush=True)

        answer_record = {
            "run_id": run_id,
            "dataset": dataset,
            "sample_no": sample_no,
            "dataset_index": idx,
            "query": query,
            "label": label,
            "final_answer": final_answer,
            "final_response": final_response,
            "predict_answer": predict_answer,
            "is_correct": is_correct,
            "answer_mode": "direct_llm",
            "fallback_reason": fallback_reason,
            "num_candidate_paths": 0,
            "num_relevant_paths": 0,
        }
        record_stage(records, "answers", answer_record, stage_writer)
        return {
            "sample_no": sample_no,
            "dataset_index": idx,
            "records": records,
            "completed": True,
            "skipped": False,
            "is_correct": is_correct,
            "final_answer": final_answer,
        }

    if len(kgs_with_entities) >= 2:
        aligned_triplets = align_entity_graphs_pairwise(
            umls_entities,
            primekg_entities,
            hetionet_entities,
            model_umls,
            pair_workers=args.alignment_workers,
            max_pairs_per_entity=args.max_alignment_pairs_per_entity,
        )
        alignment_skipped = False
        alignment_skip_reason = None
    else:
        aligned_triplets = []
        alignment_skipped = True
        alignment_skip_reason = "Fewer than two KGs returned entities"
    print(f"[sample {sample_no}] graph_aligner done", flush=True)
    graph_record = {
        "run_id": run_id,
        "dataset": dataset,
        "sample_no": sample_no,
        "dataset_index": idx,
        "aligned_triplets": aligned_triplets,
        "entity_counts": entity_counts,
        "kgs_with_entities": kgs_with_entities,
    }
    if alignment_skipped:
        graph_record["skipped"] = True
        graph_record["reason"] = alignment_skip_reason
    record_stage(records, "graph_aligner", graph_record, stage_writer)

    chains_all = {}
    if umls_result and umls_result.get("chains"):
        chains_all[kg_names["umls"]] = umls_result.get("chains")
    if primekg_result and primekg_result.get("chains"):
        chains_all[kg_names["primekg"]] = primekg_result.get("chains")
    if hetionet_result and hetionet_result.get("chains"):
        chains_all[kg_names["hetionet"]] = hetionet_result.get("chains")

    fuser = GraphFuser(chains_per_kg=chains_all, aligned_entities=aligned_triplets)

    max_candidate_paths = getattr(args, "max_candidate_paths", None)
    candidate_paths = fuser.get_all_candidate_paths(topic_ent, limit=max_candidate_paths)
    raw_candidate_path_count = len(candidate_paths)
    candidate_path_limit_hit = bool(
        max_candidate_paths and max_candidate_paths > 0 and len(candidate_paths) >= max_candidate_paths
    )
    if candidate_path_limit_hit:
        print(f"[sample {sample_no}] candidate paths {len(candidate_paths)} limit_hit=True", flush=True)
    else:
        print(f"[sample {sample_no}] candidate paths {raw_candidate_path_count}", flush=True)

    paths_with_desc = fuser.generate_path_descriptions(
        candidate_paths,
        model_umls,
        workers=args.path_workers,
    )

    relevant_paths = fuser.filter_paths_by_relevance(
        query,
        paths_with_desc,
        model_umls,
        workers=args.path_workers,
    )
    print(f"[sample {sample_no}] contextual_pruner done relevant={len(relevant_paths)}", flush=True)
    context_record = {
        "run_id": run_id,
        "dataset": dataset,
        "sample_no": sample_no,
        "dataset_index": idx,
        "topic_ent": topic_ent,
        "candidate_paths": candidate_paths,
        "paths_with_desc": paths_with_desc,
        "relevant_paths": relevant_paths,
        "raw_candidate_path_count": raw_candidate_path_count,
        "candidate_path_limit_hit": candidate_path_limit_hit,
    }
    record_stage(records, "contextual_pruner", context_record, stage_writer)

    final_answer, final_response = predict_final_answer(query, relevant_paths, model_umls)

    predict_answer = final_answer if final_answer in {"A", "B", "C", "D"} else "None"

    is_correct = predict_answer == label
    print(f"[sample {sample_no}] final answer done", flush=True)

    answer_record = {
        "run_id": run_id,
        "dataset": dataset,
        "sample_no": sample_no,
        "dataset_index": idx,
        "query": query,
        "label": label,
        "final_answer": final_answer,
        "final_response": final_response,
        "predict_answer": predict_answer,
        "is_correct": is_correct,
        "answer_mode": "kgflow",
        "num_candidate_paths": len(candidate_paths),
        "raw_candidate_path_count": raw_candidate_path_count,
        "candidate_path_limit_hit": candidate_path_limit_hit,
        "num_relevant_paths": len(relevant_paths),
    }
    record_stage(records, "answers", answer_record, stage_writer)

    return {
        "sample_no": sample_no,
        "dataset_index": idx,
        "records": records,
        "completed": True,
        "skipped": False,
        "is_correct": is_correct,
        "final_answer": final_answer,
    }

def write_sample_records(stage_paths, sample_result):
    for stage_name in ("knowledge_explorer", "graph_aligner", "contextual_pruner", "answers"):
        for record in sample_result["records"].get(stage_name, []):
            append_jsonl(stage_paths[stage_name], record)

def write_sample_records_with_writer(stage_writer, sample_result):
    for stage_name in ("knowledge_explorer", "graph_aligner", "contextual_pruner", "answers"):
        for record in sample_result["records"].get(stage_name, []):
            stage_writer.append(stage_name, record)

def make_sample_error_result(sample_no, idx, dataset, run_id, exc):
    return {
        "sample_no": sample_no,
        "dataset_index": idx,
        "records": {
            "knowledge_explorer": [],
            "graph_aligner": [],
            "contextual_pruner": [],
            "answers": [{
                "run_id": run_id,
                "dataset": dataset,
                "sample_no": sample_no,
                "dataset_index": idx,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }],
        },
        "completed": False,
        "skipped": False,
        "errored": True,
        "is_correct": False,
        "final_answer": None,
    }

def run_dataset_samples(dataset, data, sample_indices, args, stage_paths, tasks_to_run, kg_names, model_umls, run_id):
    sample_results = []
    stage_writer = StageWriter(stage_paths)

    sample_no_offset = getattr(args, "sample_no_offset", 0) or 0

    if args.sample_workers == 1:
        iterator = enumerate(sample_indices, start=sample_no_offset)
        for sample_no, idx in tqdm(iterator, total=len(sample_indices), desc=f'Evaluating {dataset}'):
            try:
                result = process_sample(
                    sample_no,
                    idx,
                    dataset,
                    data,
                    args,
                    tasks_to_run,
                    kg_names,
                    model_umls,
                    run_id,
                    stage_writer,
                )
            except Exception as exc:
                result = make_sample_error_result(sample_no, idx, dataset, run_id, exc)
                write_sample_records_with_writer(stage_writer, result)
            sample_results.append(result)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.sample_workers) as executor:
            future_to_sample = {
                executor.submit(
                    process_sample,
                    sample_no,
                    idx,
                    dataset,
                    data,
                    args,
                    tasks_to_run,
                    kg_names,
                    model_umls,
                    run_id,
                    stage_writer,
                ): (sample_no, idx)
                for sample_no, idx in enumerate(sample_indices, start=sample_no_offset)
            }

            for future in tqdm(
                concurrent.futures.as_completed(future_to_sample),
                total=len(future_to_sample),
                desc=f'Evaluating {dataset}',
            ):
                sample_no, idx = future_to_sample[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = make_sample_error_result(sample_no, idx, dataset, run_id, exc)
                    write_sample_records_with_writer(stage_writer, result)
                sample_results.append(result)

    sample_results.sort(key=lambda item: item["sample_no"])
    accurate_sample_idx = [
        result["dataset_index"]
        for result in sample_results
        if result.get("completed") and result.get("is_correct")
    ]
    response_all = [
        result["final_answer"]
        for result in sample_results
        if result.get("completed")
    ]
    skipped_sample_idx = [
        result["dataset_index"]
        for result in sample_results
        if result.get("skipped")
    ]
    errored_sample_idx = [
        result["dataset_index"]
        for result in sample_results
        if result.get("errored")
    ]

    return accurate_sample_idx, response_all, skipped_sample_idx, errored_sample_idx

if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument("--model", type=str, default=DEFAULT_MODEL)
    arg.add_argument("--max_token", type=int, default=2048)
    arg.add_argument("--N", type=int, default=5)
    arg.add_argument("--w1", type=float, default=0.5)
    arg.add_argument("--w2", type=float, default=0.5)
    arg.add_argument("--max_tries", type=int, default=6)
    arg.add_argument("--max_hop", type=int, default=3)
    arg.add_argument("--max_entities", type=int, default=None)
    arg.add_argument("--max_entity_candidates", type=int, default=None)
    arg.add_argument("--max_candidate_paths", type=int, default=None)
    arg.add_argument("--max_alignment_entities", type=int, default=None)
    arg.add_argument("--max_alignment_pairs_per_entity", type=int, default=None)
    arg.add_argument("--entity_description_workers", type=int, default=1)
    arg.add_argument("--skip_entity_descriptions", action="store_true")
    arg.add_argument("--alignment_workers", type=int, default=1)
    arg.add_argument("--path_workers", type=int, default=1)
    arg.add_argument("--kg_worker_timeout", type=float, default=None)
    arg.add_argument("--umls_workers", type=int, default=1)
    arg.add_argument("--primekg_workers", type=int, default=1)
    arg.add_argument("--hetionet_workers", type=int, default=1)
    arg.add_argument("--datasets", type=str, nargs='+', default=["mmlu", "medqa","medmcqa","pubmedqa","bioasq"])
    arg.add_argument("--sample_size", type=int, default=None)
    arg.add_argument("--sample_seed", type=int, default=42)
    arg.add_argument("--sample_indices_file", type=str, default=None)
    arg.add_argument("--sample_no_offset", type=int, default=0)
    arg.add_argument("--sample_workers", type=int, default=1)
    arg.add_argument("--output_dir", type=str, default=None)
    arg.add_argument("--primekg_device", type=str, default="cuda:1")
    arg.add_argument("--hetionet_device", type=str, default="cuda:2")
    arg.add_argument("--umls_device", type=str, default="cuda:0")
    arg.add_argument("--umls_url", type=str, required=True)
    arg.add_argument("--umls_username", type=str, default="neo4j")
    arg.add_argument("--umls_password", type=str, required=True)
    arg.add_argument("--primekg_url", type=str, required=True)
    arg.add_argument("--primekg_username", type=str, default="neo4j")
    arg.add_argument("--primekg_password", type=str, required=True)
    arg.add_argument("--primekg_database", type=str, default=None)
    arg.add_argument("--hetionet_url", type=str, required=True)
    arg.add_argument("--hetionet_username", type=str, default="neo4j")
    arg.add_argument("--hetionet_password", type=str, required=True)


    args = arg.parse_args()
    if args.sample_workers < 1:
        arg.error("--sample_workers must be >= 1")
    if args.sample_no_offset < 0:
        arg.error("--sample_no_offset must be >= 0")
    for positive_arg in ("entity_description_workers", "alignment_workers", "path_workers"):
        if getattr(args, positive_arg) < 1:
            arg.error(f"--{positive_arg} must be >= 1")
    for positive_arg in ("umls_workers", "primekg_workers", "hetionet_workers"):
        if getattr(args, positive_arg) < 1:
            arg.error(f"--{positive_arg} must be >= 1")
    if args.kg_worker_timeout is not None and args.kg_worker_timeout <= 0:
        arg.error("--kg_worker_timeout must be > 0")
    if any(getattr(args, worker_arg) != 1 for worker_arg in ("umls_workers", "primekg_workers", "hetionet_workers")):
        arg.error("Local model workers must each be 1 because every worker loads a model")
    try:
        worker_visible_devices = [
            resolve_worker_cuda_visible_device(args.umls_device),
            resolve_worker_cuda_visible_device(args.primekg_device),
            resolve_worker_cuda_visible_device(args.hetionet_device),
        ]
    except ValueError as exc:
        arg.error(str(exc))
    if len(set(worker_visible_devices)) != len(worker_visible_devices):
        arg.error("KG worker devices must resolve to three distinct GPUs")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_dir or os.path.join("runs", f"kgflow_{run_id}")
    os.makedirs(output_root, exist_ok=True)
    write_json(os.path.join(output_root, "args.json"), vars(args))

    workers = {}
    try:
        workers, worker_info = start_kg_subprocess_workers(args, output_root)
        tasks_to_run = [
            (worker_info["umls"]["name"], workers["umls"]),
            (worker_info["primekg"]["name"], workers["primekg"]),
            (worker_info["hetionet"]["name"], workers["hetionet"]),
        ]
        kg_names = {
            "umls": worker_info["umls"]["name"],
            "primekg": worker_info["primekg"]["name"],
            "hetionet": worker_info["hetionet"]["name"],
        }
        model_umls = WorkerModelProxy(workers["umls"])
        datasets = args.datasets

        for dataset in datasets:
            dataset_output_dir = os.path.join(output_root, dataset)
            os.makedirs(dataset_output_dir, exist_ok=True)
            stage_paths = {
                "knowledge_explorer": os.path.join(dataset_output_dir, "knowledge_explorer.jsonl"),
                "graph_aligner": os.path.join(dataset_output_dir, "graph_aligner.jsonl"),
                "contextual_pruner": os.path.join(dataset_output_dir, "contextual_pruner.jsonl"),
                "answers": os.path.join(dataset_output_dir, "answers.jsonl"),
                "sample_indices": os.path.join(dataset_output_dir, "sample_indices.json"),
                "summary": os.path.join(dataset_output_dir, "summary.json"),
            }

            if "pubmedqa" in dataset:
                data = PubMedQADataset()
            else:
                data = QADataset(dataset)

            if args.sample_indices_file:
                sample_indices = load_sample_indices(args.sample_indices_file)
            else:
                sample_indices = select_indices(len(data), args.sample_size, args.sample_seed)
            write_json(stage_paths["sample_indices"], {
                "dataset": dataset,
                "dataset_size": len(data),
                "sample_size": len(sample_indices),
                "sample_seed": args.sample_seed,
                "sample_indices_file": args.sample_indices_file,
                "sample_no_offset": args.sample_no_offset,
                "indices": sample_indices,
            })

            accurate_sample_idx, response_all, skipped_sample_idx, errored_sample_idx = run_dataset_samples(
                dataset,
                data,
                sample_indices,
                args,
                stage_paths,
                tasks_to_run,
                kg_names,
                model_umls,
                run_id,
            )

            summary = {
                "run_id": run_id,
                "dataset": dataset,
                "dataset_size": len(data),
                "sample_size": len(sample_indices),
                "sample_seed": args.sample_seed,
                "sample_indices_file": args.sample_indices_file,
                "sample_no_offset": args.sample_no_offset,
                "sample_workers": args.sample_workers,
                "kg_worker_mode": "subprocess_pool_per_kg",
                "kg_worker_counts": {
                    "umls": args.umls_workers,
                    "primekg": args.primekg_workers,
                    "hetionet": args.hetionet_workers,
                },
                "kg_workers": worker_info,
                "max_candidate_paths": args.max_candidate_paths,
                "max_alignment_entities": args.max_alignment_entities,
                "max_alignment_pairs_per_entity": args.max_alignment_pairs_per_entity,
                "entity_description_workers": args.entity_description_workers,
                "alignment_workers": args.alignment_workers,
                "path_workers": args.path_workers,
                "kg_worker_timeout": args.kg_worker_timeout,
                "skip_entity_descriptions": args.skip_entity_descriptions,
                "completed_predictions": len(response_all),
                "skipped": len(skipped_sample_idx),
                "errors": len(errored_sample_idx),
                "correct": len(accurate_sample_idx),
                "accuracy": (len(accurate_sample_idx) / len(response_all)) if response_all else None,
                "correct_indices": accurate_sample_idx,
                "skipped_indices": skipped_sample_idx,
                "error_indices": errored_sample_idx,
                "stage_files": stage_paths,
            }
            write_json(stage_paths["summary"], summary)

            if len(response_all) > 0:
                accuracy = len(accurate_sample_idx) / len(response_all)
                print(f"Dataset:{dataset}, ACC:{accuracy:.4f}, Correct:{len(accurate_sample_idx)}/{len(response_all)}")
            else:
                print(f"Dataset:{dataset}, No valid responses generated")
    finally:
        close_kg_subprocess_workers(workers)



    
    
