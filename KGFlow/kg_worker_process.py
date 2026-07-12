import argparse
from dataclasses import asdict, is_dataclass
import json
import os
from types import SimpleNamespace
import sys
import time
import traceback


DEFAULT_MODEL = "llama3.1-8b"


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


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_db(kg_key, args):
    if kg_key == "umls":
        from KGs.umls import UMLSDatabase

        return UMLSDatabase(args.umls_url, args.umls_username, args.umls_password, "umls")
    if kg_key == "primekg":
        from KGs.primeKG import PrimeDatabase

        return PrimeDatabase(
            args.primekg_url,
            args.primekg_username,
            args.primekg_password,
            "primeKG",
            database=args.primekg_database,
        )
    if kg_key == "hetionet":
        from KGs.hetionet import HetionetDatabase

        return HetionetDatabase(args.hetionet_url, args.hetionet_username, args.hetionet_password, "hetionet")
    raise ValueError(f"Unsupported KG key: {kg_key}")


def build_model(args):
    from local_llm import ChatModel

    return ChatModel(args.model, args.model, max_token=args.max_token, device="cuda:0")

def extract_entities_from_saved_chains(chains, base_args, db, model, sample_no=None):
    from kg_alignment import extract_entities_from_chains, has_extractable_chains

    if not has_extractable_chains(chains):
        return []

    worker_args = SimpleNamespace(**base_args)
    kg_name = db.get_name()
    prefix = f"[sample {sample_no}][{kg_name}]"
    extract_started_at = time.monotonic()
    entities = extract_entities_from_chains(
        chains,
        model,
        db,
        max_entities=worker_args.max_alignment_entities,
        description_workers=worker_args.entity_description_workers,
        describe_entities=not getattr(worker_args, "skip_entity_descriptions", False),
    )
    print(
        f"{prefix} entity extraction done entities={len(entities)} "
        f"elapsed={time.monotonic() - extract_started_at:.1f}s",
        file=sys.stderr,
        flush=True,
    )
    return entities

def run_knowledge_explorer(request, base_args, db, model):
    from kg_alignment import has_extractable_chains
    from kg_explore import process_query

    worker_args = SimpleNamespace(**base_args)
    kg_worker_timeout = getattr(worker_args, "kg_worker_timeout", None)
    if kg_worker_timeout and kg_worker_timeout > 0:
        worker_args._kgflow_deadline = time.monotonic() + kg_worker_timeout

    query = request["query"]
    sample_no = request.get("sample_no")
    kg_name = db.get_name()
    prefix = f"[sample {sample_no}][{kg_name}]"

    started_at = time.monotonic()
    print(f"{prefix} process_query start", file=sys.stderr, flush=True)
    success, topic_ent, chains = process_query(query, worker_args, db, model)
    print(
        f"{prefix} process_query done success={success} topic_entities={len(topic_ent)} "
        f"hops={len(chains)} elapsed={time.monotonic() - started_at:.1f}s",
        file=sys.stderr,
        flush=True,
    )

    entities = []
    if has_extractable_chains(chains):
        entities = extract_entities_from_saved_chains(chains, base_args, db, model, sample_no)

    return {
        "success": success,
        "topic_ent": topic_ent,
        "chains": chains,
        "entities": entities,
        "error": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    protocol_stdout = sys.stdout
    sys.stdout = sys.stderr

    config = load_config(args.config)
    kg_key = config["kg_key"]
    base_args = config["args"]
    runtime_args = SimpleNamespace(**base_args)

    print(
        f"[worker {kg_key}] starting pid={os.getpid()} visible_gpus={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        file=sys.stderr,
        flush=True,
    )
    db = build_db(kg_key, runtime_args)
    model = build_model(runtime_args)
    print(f"[worker {kg_key}] ready name={db.get_name()}", file=sys.stderr, flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = {}
        try:
            request = json.loads(line)
            request_id = request.get("id")
            request_type = request.get("type")
            if request_type == "shutdown":
                response = {"id": request_id, "ok": True, "result": "shutdown"}
                protocol_stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                protocol_stdout.flush()
                break
            if request_type == "process_query":
                result = run_knowledge_explorer(request, base_args, db, model)
            elif request_type == "extract_entities":
                result = extract_entities_from_saved_chains(
                    request.get("chains"),
                    base_args,
                    db,
                    model,
                    request.get("sample_no"),
                )
            elif request_type == "generate":
                result = model.generate_response(request["prompt"], request.get("temperature", 1.0))
            else:
                raise ValueError(f"Unknown request type: {request_type}")
            response = {"id": request_id, "ok": True, "result": make_jsonable(result)}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = {
                "id": request.get("id"),
                "ok": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }

        protocol_stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        protocol_stdout.flush()

    try:
        db.close()
    except Exception:
        pass
    print(f"[worker {kg_key}] stopped", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
