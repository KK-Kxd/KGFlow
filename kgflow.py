from kg_explore import process_query
from kg_alignment import extract_entities_from_chains, align_entity_graphs_pairwise
from KGs.primeKG import PrimeDatabase
from KGs.hetionet import HetionetDatabase
from KGs.umls import UMLSDatabase
from KGs.primeKG import PrimeDatabase
from local_llm import ChatModel
from kg_fuser import GraphFuser
from Dataset.QADataset import QADataset
from Dataset.PubMedQADataset import PubMedQADataset
from tqdm import tqdm
import concurrent.futures
import argparse
import re
import torch

from typing import List, Tuple, Any

torch.manual_seed(42)

def predict_final_answer(query_with_options: str, relevant_paths_with_descriptions: List[Tuple[List[Tuple], str]], model: Any) -> str:
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
        
        match = re.search(r"Answer:\s*([A-D])", response, re.IGNORECASE)
        if match:
            final_answer = match.group(1).upper()
            return final_answer
        else:
            return "Parsing Failed"

def process_and_extract_worker(query, args, db_instance, model_instance):
    success, topic_ent, chains = process_query(query, args, db_instance, model_instance)

    if not success:
        return success, topic_ent, chains, None 

    e_l = extract_entities_from_chains(chains, model_instance, db_instance)
    return success, topic_ent, chains, e_l

if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument("--model", type=str, default="llama3.1-8b")
    arg.add_argument("--N", type=int, default=5)
    arg.add_argument("--w1", type=float, default=0.5)
    arg.add_argument("--w2", type=float, default=0.5)
    arg.add_argument("--max_tries", type=int, default=6)
    arg.add_argument("--max_hop", type=int, default=3)
    arg.add_argument("--datasets", type=str, nargs='+', default=["mmlu", "medqa","medmcqa","pubmedqa","bioasq"])
    arg.add_argument("--umls_url", type=str, default="bolt://localhost:7001")
    arg.add_argument("--umls_username", type=str, default="neo4j")
    arg.add_argument("--umls_password", type=str, default="password")
    arg.add_argument("--primekg_url", type=str, default="bolt://localhost:7002")
    arg.add_argument("--primekg_username", type=str, default="neo4j")
    arg.add_argument("--primekg_password", type=str, default="password")
    arg.add_argument("--hetionet_url", type=str, default="bolt://localhost:7003")
    arg.add_argument("--hetionet_username", type=str, default="neo4j")
    arg.add_argument("--hetionet_password", type=str, default="password")


    args = arg.parse_args()
    primeKG = PrimeDatabase(args.primekg_url,args.primekg_username, args.primekg_password, "prime")
    model_primeKG = ChatModel(args.model, args.model, "cuda:1")
    hetionet = HetionetDatabase(args.hetionet_url,args.hetionet_username, args.hetionet_password)
    model_hetionet = ChatModel(args.model, args.model, "cuda:2")
    umls = UMLSDatabase(args.umls_url,args.umls_username, args.umls_password)
    model_umls = ChatModel(args.model, args.model)
    tasks_to_run = [
            ("UMLS", umls, model_umls),
            ("PrimeKG", primeKG, model_primeKG),
            ("Hetionet", hetionet, model_hetionet)
            ]
    datasets = args.datasets

    for dataset in datasets:
        if "pubmedqa" in dataset:
            data = PubMedQADataset()
        else:
            data = QADataset(dataset)

        accurate_sample_idx = []
        response_all = []

        for idx, d in tqdm(enumerate(data), total=len(data), desc=f'Evaluating data'):
            if 'text' in d and 'answer' in d:
                query = d['text']
                label = d['answer']

            all_results = {}

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks_to_run)) as executor:
                future_to_name = {
                    executor.submit(process_and_extract_worker, query, args, db, model): name
                    for name, db, model in tasks_to_run
                }

                for future in concurrent.futures.as_completed(future_to_name):
                    name = future_to_name[future]
                    try:
                        success, topic_ent, chains, e_l = future.result()
                        all_results[name] = {
                            "success": success, 
                            "topic_ent": topic_ent, 
                            "chains": chains, 
                            "entities": e_l,
                            "error": None
                        }
                    except Exception as exc:
                        all_results[name] = {"error": exc}

            umls_result = all_results.get("UMLS")
            primekg_result = all_results.get("PrimeKG")
            hetionet_result = all_results.get("Hetionet")

            # Only use available results
            umls_entities = umls_result.get("entities") if umls_result else None
            primekg_entities = primekg_result.get("entities") if primekg_result else None
            hetionet_entities = hetionet_result.get("entities") if hetionet_result else None

            aligned_triplets = align_entity_graphs_pairwise(umls_entities, primekg_entities, hetionet_entities, model_umls)

            chains_all = {}
            if umls_result and umls_result.get("chains"):
                chains_all['umls'] = umls_result.get("chains")
            if primekg_result and primekg_result.get("chains"):
                chains_all['primeKG'] = primekg_result.get("chains")
            if hetionet_result and hetionet_result.get("chains"):
                chains_all['hetionet'] = hetionet_result.get("chains")

            fuser = GraphFuser(chains_per_kg=chains_all, aligned_entities=aligned_triplets)

            # Get topic_ent from available results
            topic_ent = None
            if primekg_result and primekg_result.get("topic_ent"):
                topic_ent = primekg_result.get("topic_ent")
            elif umls_result and umls_result.get("topic_ent"):
                topic_ent = umls_result.get("topic_ent")
            elif hetionet_result and hetionet_result.get("topic_ent"):
                topic_ent = hetionet_result.get("topic_ent")

            if not topic_ent:
                print(f"Warning: No topic entity found for query: {query[:50]}...")
                continue

            candidate_paths = fuser.get_all_candidate_paths(topic_ent)

            paths_with_desc = fuser.generate_path_descriptions(candidate_paths, model_umls)

            relevant_paths = fuser.filter_paths_by_relevance(query, paths_with_desc, model_umls)

            final_answer = predict_final_answer(query, relevant_paths, model_umls)

            predict_answer = 'None'
            matches = re.findall(r'Answer:\s*([A-D])\b', final_answer, re.IGNORECASE)
            if matches:
                predict_answer = matches[-1].upper()
            elif label in final_answer:
                predict_answer = label
            else:
                predict_answer = "None"  

            is_correct = predict_answer == label

            if is_correct:
                accurate_sample_idx.append(idx)
            response_all.append(final_answer)

        if len(response_all) > 0:
            accuracy = len(accurate_sample_idx) / len(response_all)
            print(f"Dataset:{dataset}, ACC:{accuracy:.4f}, Correct:{len(accurate_sample_idx)}/{len(response_all)}")
        else:
            print(f"Dataset:{dataset}, No valid responses generated")



    
    

