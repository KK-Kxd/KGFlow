from local_llm import ChatModel
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re
from typing import List, Dict, Set, Tuple, Optional

@dataclass
class EntityInfo:
    ent_id: str  # CUI/name
    name: str
    type: str
    neighbors: set 
    source: str
    desc: Optional[str] = None

    def __post_init__(self):
        if self.neighbors is None:
            self.neighbors = set()
    def get_tuple_id(self) -> Tuple[str, str]:
        return (self.source, self.name)

def extract_entities_from_chains(chains, model, db):
    """
    Returns:
      {'ent_id': ent_id, 'name': 'XXXX','ent_type': ent_type, 'neighbors': {'(h1, r1, XXXX)', '(XXXX, r2, t2)'}, 'desc': "description"}
    """
    entities = {}  # key: entity_name, value: EntityInfo

    if not chains or chains == [[]]:
        return entities

    for hop in chains:
        if not hop:  
            continue
        for triplet in hop:
            if len(triplet) >= 3:
                head_entity = triplet[0]
                relation = triplet[1]
                tail_entity = triplet[2]

                if isinstance(head_entity, dict):
                    head_name = head_entity.get('name', '')
                    head_id = head_entity.get('cui', '')
                    if head_id in ['', 'unknown', 'None', None]:
                        head_id = head_name 
                else:
                    head_name = str(head_entity)
                    head_id = head_name


                if isinstance(tail_entity, dict):
                    tail_name = tail_entity.get('name', '')
                    tail_id = tail_entity.get('cui', '')
                    if tail_id in ['', 'unknown', 'None', None]:
                        tail_id = tail_name  
                else:
                    tail_name = str(tail_entity)
                    tail_id = tail_name

                if isinstance(relation, dict):
                    relation_name = relation.get('type', relation.get('relation', str(relation)))
                else:
                    relation_name = str(relation)

                if head_name and head_name.strip():
                    if head_name not in entities:
                        entities[head_name] = EntityInfo(
                            ent_id=head_id,
                            name=head_name,
                            neighbors=set(),
                            source=db.db_name()
                        )
                    entities[head_name].neighbors.add(f"({head_name}, {relation_name}, {tail_name})")

                if tail_name and tail_name.strip():
                    if tail_name not in entities:
                        entities[tail_name] = EntityInfo(
                            ent_id=tail_id,
                            name=tail_name,
                            neighbors=set()
                        )
                    entities[tail_name].neighbors.add(f"({head_name}, {relation_name}, {tail_name})")

    for entity_name, entity_info in entities.items():
        try:
            entity_info.desc = get_entity_description(entity_name, entity_info.neighbors, db, model)
        except Exception as e:
            entity_info.desc = "No description available"

    return entities



def generate_medical_entity_prompt(entity: Dict[str, str], neighbors: List[List[str]]) -> str:

    
    formatted_neighbors = []
    for neighbor in neighbors:
        if len(neighbor) >= 3:
            formatted_neighbors.append(f"({neighbor[0]}, {neighbor[1]}, {neighbor[2]})")

    neighbors_str = ", ".join(formatted_neighbors) if formatted_neighbors else "No related knowledge tuples available"

    prompt = f"""You are a medical expert. Your task is to provide a concise, professional medical description for the given medical entity based on:
1. Your medical knowledge
2. The provided knowledge tuples from medical knowledge graphs

Requirements:
- Provide a clear, accurate medical description in 1-2 sentences
- Use appropriate medical terminology
- Focus on the entity's medical significance, function, or clinical relevance
- Keep the description under 80 tokens
- Be precise and factual

Example:
[KNOWLEDGE]: Given [Entity] Myocardial Infarction and its related [Knowledge Tuples]: [(Myocardial Infarction, causes, Chest Pain), (Myocardial Infarction, treated_by, Aspirin), (Coronary Artery Disease, may_cause, Myocardial Infarction)].
[Input]: What is Myocardial Infarction? Please provide a medical description based on your knowledge and the knowledge tuples.
[Output]: Myocardial Infarction is a serious cardiac condition caused by blocked blood flow to heart muscle, commonly presenting with chest pain and treated with medications like aspirin.

Now please answer:
[KNOWLEDGE]: Given [Entity] {entity['name']} and its related [Knowledge Tuples]: [{neighbors_str}].
[Input]: What is {entity['name']}? Please provide a medical description based on your knowledge and the knowledge tuples.
[Output]: """

    return prompt

def get_entity_description(entity, neighbors,db, model) -> Optional[str]:
        if neighbors and len(neighbors) < 5:
            db_neighbors = db.get_neighbors(entity['ent_id'])
            if neighbors:

                all_neighbors = neighbors + db_neighbors[:max(0, 5 - len(neighbors))]
            else:
                all_neighbors = db_neighbors
        prompt = generate_medical_entity_prompt(entity, all_neighbors)
        res = model.generate_response(prompt, 500, 0.4)
        return res

def format_alignment_prompt(main_entity: EntityInfo, candidate_entity: EntityInfo, k_threshold: int = 18) -> str:
    main_tuples_str = ", ".join([f"({main_entity.name}, {rel}, {neighbor})" for rel, neighbor in main_entity.neighbors])
    candidate_tuples_str = ", ".join([f"({candidate_entity.name}, {rel}, {neighbor})" for rel, neighbor in candidate_entity.neighbors])

    prompt = f"""You are an expert in medical entity alignment. Your task is to determine if two medical entities from different knowledge
graphs refer to the same real-world medical concept.
Now given [Main Entity] le = Entity({{ Name: ”{main_entity.name}”, Type: ”{main_entity.type}”, Description: ”{main_entity.desc or ''}”, Structure: [{main_tuples_str}] }}), and [Candidate Entity] re = Entity({{ Name: ”{candidate_entity.name}”, Type: ”{candidate_entity.type}”, Description: ”{candidate_entity.desc or ''}”, Structure: [{candidate_tuples_str}] }}).
Do [Main Entity] and [Candidate Entity] align or match? Think of the answer STEP BY STEP with name, type,
description, structure, YOUR OWN KNOWLEDGE:
Step 1, think of [NAME SIMILARITY] = A out of 10, using entity name and entity type and YOUR OWN KNOWLEDGE of medical terminology, synonyms, and abbreviations.
Step 2, think of [PROBABILITY OF DESCRIPTION POINTING SAME ENTITY] = B out of 10, using the provided descriptions and YOUR OWN KNOWLEDGE of medical concepts.
Step 3, think of [STRUCTURE SIMILARITY] = C out of 10, using the knowledge tuples (relationships) and entity type and YOUR OWN KNOWLEDGE of medical entity relationships.
NOTICE, the information provided above may not be sufficient, so use YOUR OWN KNOWLEDGE of medical terminology, anatomy, diseases, treatments, and relationships to complete the analysis.
Output answer strictly in format:
[NAME SIMILARITY] = A out of 10
[PROBABILITY OF DESCRIPTION POINTING SAME ENTITY] = B out of 10
[STRUCTURE SIMILARITY] = C out of 10
[FINAL DECISION] = YES/NO (YES if A+B+C >= {k_threshold}, NO otherwise)"""
    
    return prompt

def check_alignment(main_entity: EntityInfo, candidate_entity: EntityInfo, model) -> bool:

    prompt = format_alignment_prompt(main_entity, candidate_entity)

    response = model.generate_response(prompt, 256, 0.4)

    match = re.search(r"\[FINAL DECISION\] = (YES|NO)", response)
    if match:
        return match.group(1) == "YES"
    
    return False


def _align_pairwise(
    entities1: List[EntityInfo],
    entities2: List[EntityInfo],
    model
) -> List[Tuple[EntityInfo, EntityInfo]]:

    aligned_pairs = []

    # Handle None inputs
    if entities1 is None or entities2 is None:
        return aligned_pairs

    aligned_ent2_ids = set()


    for ent1 in entities1:
        for ent2 in entities2:
            if ent2.ent_id in aligned_ent2_ids:
                continue
            
            if check_alignment(ent1, ent2, model):
                print(f"  Found alignment: ({ent1}) <-> ({ent2})")
                aligned_pairs.append((ent1, ent2))
                aligned_ent2_ids.add(ent2.ent_id)
                break 
                
    return aligned_pairs


def align_entity_graphs_pairwise(
    umls_entities: List[EntityInfo],
    primekg_entities: List[EntityInfo],
    hetionet_entities: List[EntityInfo],
    model
) -> List[Tuple[EntityInfo, EntityInfo]]:
    all_aligned_pairs = []

    umls_primekg_pairs = _align_pairwise(umls_entities, primekg_entities, model)
    all_aligned_pairs.extend(umls_primekg_pairs)

    umls_hetionet_pairs = _align_pairwise(umls_entities, hetionet_entities, model)
    all_aligned_pairs.extend(umls_hetionet_pairs)
    
    primekg_hetionet_pairs = _align_pairwise(primekg_entities, hetionet_entities, model)
    all_aligned_pairs.extend(primekg_hetionet_pairs)
    
    return all_aligned_pairs
