from ollama import chat
from ollama import ChatResponse
import asyncio
from ollama import AsyncClient
import promptTemplate
import json
import os

import argparse 
import re
from tqdm import tqdm
import logging
from datetime import datetime

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
    if response.strip()[-1] != '}':
        response += '}'
    try:
        data = parse_and_fix_json(response)

        data = json.loads(response)


        validate_response(data)


        return data["entities"]

    except (json.JSONDecodeError, ValueError) as e:
        pass


def chat_llm(model, query):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query},
    ]
    response: ChatResponse = chat(model=model, messages=messages, options={"temperature": 0.4})
    return response.message.content
    
def extract_entity(query, args):
    messages = promptTemplate.entity_extract_prompt.format(query=query)
    
    for attempt in range(args.max_tries):
        response = chat_llm(args.model, messages)
        response = response.strip()
        print(f"Attempt {attempt + 1}: {response}")
        
        try:
            parsed_data = json.loads(response)
            entity_list = parsed_data.get("medical_terminologies", [])
            return entity_list
        except (json.JSONDecodeError, KeyError) as e:
            pass

    
    return []



 

def relation_score(question, entity, out_rel, in_rel, args, model):
    for attempt in range(args.max_tries):
        prompt = promptTemplate.score_relation_prompt.format(question, entity, out_rel+in_rel)
        response = model.generate_response(prompt, 256, 0.4)

        
        if response.strip()[-1] != '}':
            response += '}'
        
        try:
            data = parse_and_fix_json(response)
            
            if data is None:
                pass
            
            # Validate data format
            if not isinstance(data, dict):
                pass
            
            if "relations" not in data:
                pass
            
            relations = data["relations"]
            if not isinstance(relations, list):
                pass
            
            for relation in relations:
                if not isinstance(relation, dict):
                    pass
                if "score" not in relation or "relation" not in relation:
                    pass
            
            modified_relations = []
            for rel in relations:
                try:
                    
                    score = float(rel["score"])
                    modified_relations.append({rel["relation"]: score})
                except (ValueError, TypeError):  
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
    prompt = promptTemplate.score_entity_prompt.format(question, relation) + "; ".join(entity_candidates) + '\nScore: '

    response = model.generate_response(prompt, 256, 0.4)


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

def get_score(query, entity, args, neo4j, model):
    out_rel, in_rel = neo4j.get_entity_relationships(entity)
    if out_rel == [] and in_rel == []:
        return []
    rel_sc, out_rel, in_rel = relation_score(query, entity, out_rel, in_rel, args, model)
    ec = []
    if(rel_sc is not None and out_rel is not None and in_rel is not None):
        rel_sc_dict = {list(d.keys())[0]: list(d.values())[0] for d in rel_sc}

        scores = []

        for rel in out_rel:
            entity_candidate = neo4j.find_tail_entity(entity, rel)
            cnt = 1
            entities = None
            if_true = False
            while cnt <= 3 and entities is None and not if_true:
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
                    total_score = calculate_score(rel_sc_dict[rel], entity_sc, args.w1, args.w2)
                    ec.append(entity_name)
                    scores.append((entity, rel, entity_name, total_score))

        for rel in in_rel:
            entity_candidate = neo4j.find_head_entity(entity, rel)
            cnt = 1
            entities = None
            if_true = False
            while cnt <= 3 and entities is None and not if_true:
                entities = entity_score(query, entity_candidate, rel, args)
                
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
                    total_score = calculate_score(rel_sc_dict[rel], entity_sc, args.w1, args.w2)
                    ec.append(entity_name)
                    scores.append((entity_name, rel, entity, total_score))

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
    prompt = promptTemplate.prompt_evaluate + question
    chain_prompt = '\n'.join([f"{entity1}, {relation}, {entity2}" for entity1, relation, entity2, score in sorted_scores])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '

    response = model.generate_response(prompt, 256, 0)

    result = extract_answer(response)
    
    if if_true(result):
        return True, response
    else:
        return False, response
    
def process_query(query, args, neo4j, model):
    entities = extract_entity(query, args)
    topic_ent = []
    for ent in entities:
        topic_ent.append(neo4j.get_name(),ent)
    iteration = 0
    success = False
    beam_width = args.N
    reasoning_chains = []

    while not success and iteration < args.max_hop:
        total_scores = []
        for entity in entities:
            sc, ec = get_score(query, entity, args, neo4j, model)
            total_scores.extend(sc)

        sorted_scores = sorted(total_scores, key=lambda x: x[3], reverse=True)
        chain_reasoning = sorted_scores[:beam_width]
        reasoning_chains.append(chain_reasoning)

        success, _ = reasoning(query, chain_reasoning, args, model)

        if not success:
            entities = ec
        iteration += 1
        print(reasoning_chains)

    return success, topic_ent, reasoning_chains





    
    
        

