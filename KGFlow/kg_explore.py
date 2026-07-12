import asyncio
import promptTemplate
import json
import os
from typing import Any

try:
    from ollama import chat
    from ollama import ChatResponse
    from ollama import AsyncClient
except ImportError:
    chat = None
    ChatResponse = Any
    AsyncClient = None

import argparse 
import re
from tqdm import tqdm
import logging
import time
from datetime import datetime


class KGWorkerTimeout(TimeoutError):
    pass


def check_worker_deadline(args):
    deadline = getattr(args, "_kgflow_deadline", None)
    if deadline is not None and time.monotonic() >= deadline:
        raise KGWorkerTimeout("KG worker timeout exceeded")

def fix_json(response):
    response = re.sub(r',\s*([}\]])', r'\1', response)

    response = re.sub(r'(?<!\\)"(?![:,}\]])', r'\"', response)

    response = re.sub(r'(?<=:)\s*([a-zA-Z0-9_]+)(?=[,}\]])', r'"\1"', response)
    
    return response

def parse_and_fix_json(response):
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        fixed_response = fix_json(response)
        try:
            return json.loads(fixed_response)
        except json.JSONDecodeError as e:
            return None


def validate_response(data):
    if "entities" not in data or not isinstance(data["entities"], list):
        pass 


def process_response(response):
    if not response or not response.strip():
        return None

    if response.strip()[-1] != '}':
        response += '}'
    try:
        data = parse_and_fix_json(response)
        if data is None:
            return None


        entities = data.get("entities")
        if not isinstance(entities, list):
            return None

        return entities

    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        pass


def chat_llm(model_name, query):
    if chat is None:
        raise ImportError("ollama is not installed; pass a ChatModel instance or install ollama.")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query},
    ]
    response: ChatResponse = chat(model=model_name, messages=messages, options={"temperature": 0.4})
    return response.message.content

def extract_entity(query, args, model=None):
    messages = promptTemplate.entity_extract_prompt.format(query=query)

    for attempt in range(args.max_tries):
        check_worker_deadline(args)
        if model is not None:
            # Use the ChatModel instance
            response = model.generate_response(messages, 0.4)
        else:
            # Fallback to ollama
            response = chat_llm(args.model, messages)
        response = response.strip()
        print(f"Attempt {attempt + 1}: {response}")
        
        try:
            parsed_data = json.loads(response)
            entity_list = parsed_data.get("medical_terminologies", [])
            unique_entities = []
            seen_entities = set()
            for entity in entity_list:
                entity_name = str(entity).strip()
                if not entity_name:
                    continue
                key = entity_name.lower()
                if key in seen_entities:
                    continue
                seen_entities.add(key)
                unique_entities.append(entity_name)

            max_entities = getattr(args, "max_entities", None)
            if max_entities and max_entities > 0:
                unique_entities = unique_entities[:max_entities]
            return unique_entities
        except (json.JSONDecodeError, KeyError) as e:
            pass

    
    return []



 

def relation_score(question, entity, out_rel, in_rel, args, model):
    for attempt in range(args.max_tries):
        check_worker_deadline(args)
        prompt = promptTemplate.score_relation_prompt.format(question, entity, out_rel+in_rel)
        response = model.generate_response(prompt, 0.4)

        
        if not response or not response.strip():
            continue

        if response.strip()[-1] != '}':
            response += '}'
        
        try:
            data = parse_and_fix_json(response)
            
            if data is None:
                continue
            
            # Validate data format
            if not isinstance(data, dict):
                continue
            
            if "relations" not in data:
                continue
            
            relations = data["relations"]
            if not isinstance(relations, list):
                continue

            valid_relations = [
                relation for relation in relations
                if isinstance(relation, dict)
                and "relation" in relation
                and "score" in relation
            ]
            if not valid_relations:
                continue

            modified_relations = []
            for rel in valid_relations:
                try:
                    
                    score = float(rel["score"])
                    modified_relations.append({rel["relation"]: score})
                except (ValueError, TypeError, KeyError):  
                    pass

           
            sorted_relations = sorted(modified_relations, key=lambda x: list(x.values())[0], reverse=True)
            sorted_relations = sorted_relations[:args.N]
            rel_sc_keys = [list(d.keys())[0] for d in sorted_relations]

            # Filter out_rel and in_rel
            filtered_out_rel = [rel for rel in out_rel if rel in rel_sc_keys]
            filtered_in_rel = [rel for rel in in_rel if rel in rel_sc_keys]
            logging.info(f"filtered_out_rel: {filtered_out_rel}, filtered_in_rel: {filtered_in_rel}")
            return sorted_relations, filtered_out_rel, filtered_in_rel

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            if attempt < args.max_tries - 1:
                print("Retrying...")
            else:
                pass
                return None, None, None


    return None, None, None

def entity_score(question, entity_candidates, relation, args, model):
    check_worker_deadline(args)
    if not entity_candidates:
        return None

    max_candidates = getattr(args, "max_entity_candidates", None)
    unique_candidates = []
    seen_candidates = set()
    for candidate in entity_candidates:
        candidate_name = entity_display_name(candidate)
        if not candidate_name:
            continue
        key = entity_identity_key(candidate)
        if key in seen_candidates:
            continue
        seen_candidates.add(key)
        unique_candidates.append(candidate_name)
        if max_candidates and max_candidates > 0 and len(unique_candidates) >= max_candidates:
            break

    if not unique_candidates:
        return None

    prompt = promptTemplate.score_entity_prompt.format(question, relation) + "; ".join(unique_candidates) + '\nScore: '

    check_worker_deadline(args)
    response = model.generate_response(prompt, 0.4)


    entities = process_response(response)
    if entities is None:
        return None
    entities_with_score = [entity for entity in entities if "score" in entity]

    if not entities_with_score:
        return None

    valid_entities = []
    for entity in entities_with_score:
        try:
            entity["score"] = float(entity["score"])  
            valid_entities.append(entity)
        except (ValueError, TypeError): 
            pass

    sorted_entities = sorted(valid_entities, key=lambda x: x["score"], reverse=True)
    return sorted_entities[:args.N]
    
def calculate_score(rel_score, entity_score, weight1, weight2):
    return rel_score * weight1 + entity_score * weight2

def resolve_entity_for_db(neo4j, entity):
    if hasattr(neo4j, "resolve_entity"):
        return neo4j.resolve_entity(entity)

    entity_name = str(entity).strip()
    return {
        "id": entity_name,
        "name": entity_name,
        "type": "unknown",
        "source": neo4j.get_name() if hasattr(neo4j, "get_name") else "",
    }

def entity_display_name(entity):
    if isinstance(entity, dict):
        value = (
            entity.get("name")
            or entity.get("node_name")
            or entity.get("preferred_name")
            or entity.get("entity")
            or entity.get("CUI")
            or entity.get("cui")
            or entity.get("id")
            or entity.get("ent_id")
            or entity.get("identifier")
            or entity.get("node_index")
        )
        return str(value).strip() if value is not None else ""
    if isinstance(entity, (list, tuple)) and len(entity) >= 2:
        return str(entity[1]).strip()
    return str(entity).strip()

def entity_identity_key(entity):
    if isinstance(entity, dict):
        for key in ("source", "CUI", "cui", "id", "ent_id", "identifier", "node_index"):
            value = entity.get(key)
            if value is not None and str(value).strip():
                return f"{key}:{str(value).strip().lower()}"
    return f"name:{entity_display_name(entity).lower()}"

def candidate_lookup_by_name(candidates):
    lookup = {}
    for candidate in candidates:
        name = entity_display_name(candidate)
        if name:
            lookup.setdefault(name.lower(), candidate)
    return lookup

def dedupe_entities(entities):
    deduped = []
    seen = set()
    for entity in entities:
        key = entity_identity_key(entity)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entity)
    return deduped

def get_score(query, entity, args, neo4j, model):
    check_worker_deadline(args)
    resolved_entity = resolve_entity_for_db(neo4j, entity)
    if not resolved_entity:
        logging.info(f"Entity not found in {neo4j.get_name()}: {entity}")
        return [], []

    entity_id = resolved_entity.get("id") or resolved_entity.get("ent_id") or resolved_entity.get("name")
    entity_name = resolved_entity.get("name") or str(entity)

    out_rel, in_rel = neo4j.get_entity_relationships(entity_id)
    if out_rel == [] and in_rel == []:
        return [], []
    rel_sc, out_rel, in_rel = relation_score(query, entity_name, out_rel, in_rel, args, model)
    ec = []
    if(rel_sc is not None and out_rel is not None and in_rel is not None):
        rel_sc_dict = {list(d.keys())[0]: list(d.values())[0] for d in rel_sc}

        scores = []

        for rel in out_rel:
            check_worker_deadline(args)
            entity_candidate = neo4j.find_tail_concepts(entity_id, rel)
            entity_candidate_by_name = candidate_lookup_by_name(entity_candidate)
            cnt = 1
            entities = None
            if_true = False
            while cnt <= 3 and entities is None and not if_true:
                check_worker_deadline(args)
                entities = entity_score(query, entity_candidate, rel, args, model)
                
                if entities is not None:
                    valid_entities = []
                    for entity_info in entities:
                        if "entity" in entity_info and "score" in entity_info:
                            valid_entities.append(entity_info)
                    entities = valid_entities if valid_entities else None
                
                if entities is None:
                    cnt += 1

            if entities is not None:
                for entity_info in entities:
                    entity_name = entity_info['entity']
                    entity_sc = entity_info['score']
                    next_entity = entity_candidate_by_name.get(entity_name.lower(), entity_name)
                    next_entity_name = entity_display_name(next_entity) or entity_name
                    total_score = calculate_score(rel_sc_dict[rel], entity_sc, args.w1, args.w2)
                    ec.append(next_entity)
                    scores.append((resolved_entity["name"], rel, next_entity_name, total_score))

        for rel in in_rel:
            check_worker_deadline(args)
            entity_candidate = neo4j.find_head_concepts(entity_id, rel)
            entity_candidate_by_name = candidate_lookup_by_name(entity_candidate)
            cnt = 1
            entities = None
            if_true = False
            while cnt <= 3 and entities is None and not if_true:
                check_worker_deadline(args)
                entities = entity_score(query, entity_candidate, rel, args, model)
                
                if entities is not None:
                    valid_entities = []
                    for entity_info in entities:
                        if "entity" in entity_info and "score" in entity_info:
                            valid_entities.append(entity_info)
                    entities = valid_entities if valid_entities else None
                
                if entities is None:
                    cnt += 1

            if entities is not None:
                for entity_info in entities:
                    entity_name = entity_info['entity']
                    entity_sc = entity_info['score']
                    next_entity = entity_candidate_by_name.get(entity_name.lower(), entity_name)
                    next_entity_name = entity_display_name(next_entity) or entity_name
                    total_score = calculate_score(rel_sc_dict[rel], entity_sc, args.w1, args.w2)
                    ec.append(next_entity)
                    scores.append((next_entity_name, rel, resolved_entity["name"], total_score))

        sorted_scores = sorted(scores, key=lambda x: x[3], reverse=True)[:args.N]

        return sorted_scores, ec
    else:
        return [], []

def extract_answer(text):
    start_index = text.find("{")
    end_index = text.find("}")
    if start_index != -1 and end_index != -1:
        return text[start_index+1:end_index].strip()
    else:
        return ""
    
def if_true(prompt):
    if prompt.lower().strip().replace(" ","")=="yes":
        return True
    return False

def reasoning(question, sorted_scores, args, model):
    check_worker_deadline(args)
    prompt = promptTemplate.prompt_evaluate + question
    chain_prompt = '\n'.join([f"{entity1}, {relation}, {entity2}" for entity1, relation, entity2, score in sorted_scores])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '

    response = model.generate_response(prompt, 0.0)

    result = extract_answer(response)
    
    if if_true(result):
        return True, response
    else:
        return False, response
    
def process_query(query, args, neo4j, model):
    check_worker_deadline(args)
    entities = extract_entity(query, args, model)
    topic_ent = []
    for ent in entities:
        resolved_entity = resolve_entity_for_db(neo4j, ent)
        if resolved_entity:
            topic_ent.append((neo4j.get_name(), resolved_entity.get("name", ent)))
    iteration = 0
    success = False
    beam_width = args.N
    reasoning_chains = []

    while not success and iteration < args.max_hop:
        check_worker_deadline(args)
        total_scores = []
        next_entities = []
        for entity in entities:
            check_worker_deadline(args)
            sc, ec = get_score(query, entity, args, neo4j, model)
            total_scores.extend(sc)
            next_entities.extend(ec)

        sorted_scores = sorted(total_scores, key=lambda x: x[3], reverse=True)
        chain_reasoning = sorted_scores[:beam_width]
        reasoning_chains.append(chain_reasoning)

        success, _ = reasoning(query, chain_reasoning, args, model)

        if not success:
            entities = dedupe_entities(next_entities)
        iteration += 1
        print(reasoning_chains)

    return success, topic_ent, reasoning_chains





    
    
        
