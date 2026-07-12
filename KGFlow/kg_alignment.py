import concurrent.futures
from dataclasses import dataclass
import re
from typing import List, Dict, Set, Tuple, Optional

@dataclass
class EntityInfo:
    ent_id: str  # CUI/name
    name: str
    type: str
    neighbors: Set[Tuple[str, str, str]]
    source: str
    desc: Optional[str] = None

    def __post_init__(self):
        if self.neighbors is None:
            self.neighbors = set()
    def get_tuple_id(self) -> Tuple[str, str]:
        return (self.source, self.name)

def _db_name(db) -> str:
    if hasattr(db, "get_name"):
        return db.get_name()
    return getattr(db, "name", "unknown")

def _entity_from_raw(raw_entity, db) -> Tuple[str, str, str]:
    entity_id = ""
    entity_name = ""
    entity_type = "unknown"

    if isinstance(raw_entity, dict):
        entity_id = (
            raw_entity.get("CUI")
            or raw_entity.get("cui")
            or raw_entity.get("node_index")
            or raw_entity.get("identifier")
            or raw_entity.get("id")
            or raw_entity.get("ent_id")
            or ""
        )
        entity_name = (
            raw_entity.get("name")
            or raw_entity.get("node_name")
            or raw_entity.get("preferred_name")
            or ""
        )
        entity_type = raw_entity.get("type") or raw_entity.get("semantic_type") or "unknown"
        lookup_value = raw_entity if entity_id or entity_name else None
    else:
        entity_name = str(raw_entity).strip()
        entity_id = entity_name
        lookup_value = entity_name

    if lookup_value and hasattr(db, "resolve_entity"):
        try:
            resolved = db.resolve_entity(lookup_value)
        except Exception:
            resolved = None
        if resolved:
            entity_id = resolved.get("id") or resolved.get("ent_id") or entity_id
            entity_name = resolved.get("name") or entity_name
            entity_type = resolved.get("type") or entity_type

    if not entity_id:
        entity_id = entity_name

    return str(entity_id), entity_name, entity_type

def _relation_name(raw_relation) -> str:
    if isinstance(raw_relation, dict):
        return raw_relation.get("type") or raw_relation.get("relation") or str(raw_relation)
    return str(raw_relation)

def has_extractable_chains(chains) -> bool:
    if not chains:
        return False
    for hop in chains:
        if not hop:
            continue
        for triplet in hop:
            if isinstance(triplet, (list, tuple)) and len(triplet) >= 3:
                return True
    return False

def extract_entities_from_chains(
    chains,
    model,
    db,
    max_entities: Optional[int] = None,
    description_workers: int = 1,
    describe_entities: bool = True,
) -> List[EntityInfo]:
    """
    Returns:
      List of EntityInfo records extracted from KG reasoning chains.
    """
    entities = {}  # key: entity_name, value: EntityInfo

    if not has_extractable_chains(chains):
        return []

    for hop in chains:
        if not hop:  
            continue
        for triplet in hop:
            if len(triplet) >= 3:
                head_entity = triplet[0]
                relation = triplet[1]
                tail_entity = triplet[2]

                head_id, head_name, head_type = _entity_from_raw(head_entity, db)
                tail_id, tail_name, tail_type = _entity_from_raw(tail_entity, db)
                relation_name = _relation_name(relation)
                source = _db_name(db)

                if head_name and head_name.strip():
                    if head_name not in entities:
                        entities[head_name] = EntityInfo(
                            ent_id=head_id,
                            name=head_name,
                            type=head_type,
                            neighbors=set(),
                            source=source
                        )
                    entities[head_name].neighbors.add((head_name, relation_name, tail_name))

                if tail_name and tail_name.strip():
                    if tail_name not in entities:
                        entities[tail_name] = EntityInfo(
                            ent_id=tail_id,
                            name=tail_name,
                            type=tail_type,
                            neighbors=set(),
                            source=source
                        )
                    entities[tail_name].neighbors.add((head_name, relation_name, tail_name))

    entity_list = list(entities.values())
    if max_entities and max_entities > 0:
        entity_list = entity_list[:max_entities]

    if not describe_entities:
        return entity_list

    def describe_entity(entity_info: EntityInfo) -> None:
        try:
            entity_info.desc = get_entity_description(entity_info, entity_info.neighbors, db, model)
        except Exception:
            entity_info.desc = "No description available"

    description_workers = max(1, description_workers or 1)
    if description_workers == 1 or len(entity_list) <= 1:
        for entity_info in entity_list:
            describe_entity(entity_info)
    else:
        workers = min(description_workers, len(entity_list))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(describe_entity, entity_info) for entity_info in entity_list]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    return entity_list


def _format_neighbor(neighbor) -> str:
    if isinstance(neighbor, str):
        return neighbor
    if isinstance(neighbor, (tuple, list)) and len(neighbor) >= 3:
        return f"({neighbor[0]}, {neighbor[1]}, {neighbor[2]})"
    return str(neighbor)

def generate_medical_entity_prompt(entity: Dict[str, str], neighbors: List[List[str]]) -> str:

    
    formatted_neighbors = [_format_neighbor(neighbor) for neighbor in neighbors]

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

def get_entity_description(entity: EntityInfo, neighbors, db, model) -> Optional[str]:
    all_neighbors = list(neighbors or [])
    if len(all_neighbors) < 5 and hasattr(db, "get_neighbors"):
        try:
            db_neighbors = db.get_neighbors(entity.ent_id or entity.name)
        except Exception:
            db_neighbors = []
        all_neighbors.extend(db_neighbors[:max(0, 5 - len(all_neighbors))])
    prompt = generate_medical_entity_prompt({"name": entity.name}, all_neighbors)
    res = model.generate_response(prompt, 0.4)
    return res

def format_alignment_prompt(main_entity: EntityInfo, candidate_entity: EntityInfo, k_threshold: int = 18) -> str:
    main_tuples_str = ", ".join([_format_neighbor(neighbor) for neighbor in main_entity.neighbors])
    candidate_tuples_str = ", ".join([_format_neighbor(neighbor) for neighbor in candidate_entity.neighbors])

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

    response = model.generate_response(prompt, 0.4)

    match = re.search(r"\[FINAL DECISION\]\s*=\s*(YES|NO)", response, re.IGNORECASE)
    if match:
        return match.group(1).upper() == "YES"
    
    return False

def _normalize_entity_name(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", " ", str(name or "").casefold()).strip()

def _is_exact_name_alignment(main_entity: EntityInfo, candidate_entity: EntityInfo) -> bool:
    main_name = _normalize_entity_name(main_entity.name)
    candidate_name = _normalize_entity_name(candidate_entity.name)
    if not main_name or not candidate_name:
        return False
    return main_name == candidate_name

def _normalize_neighbor(neighbor):
    if isinstance(neighbor, tuple):
        return neighbor
    if isinstance(neighbor, list):
        return tuple(neighbor)
    if isinstance(neighbor, dict):
        return tuple(sorted((str(key), str(value)) for key, value in neighbor.items()))
    return (str(neighbor), "", "")

def _entity_info_from_dict(entity: Dict) -> Optional[EntityInfo]:
    ent_id = entity.get("ent_id") or entity.get("id") or entity.get("CUI") or entity.get("node_index")
    name = entity.get("name") or entity.get("node_name") or entity.get("preferred_name") or ent_id
    if not name:
        return None

    neighbors = entity.get("neighbors") or []
    return EntityInfo(
        ent_id=str(ent_id or name),
        name=str(name),
        type=str(entity.get("type") or entity.get("semantic_type") or "unknown"),
        neighbors={_normalize_neighbor(neighbor) for neighbor in neighbors},
        source=str(entity.get("source") or "unknown"),
        desc=entity.get("desc"),
    )

def _normalize_entities(entities) -> List[EntityInfo]:
    if entities is None:
        return []
    if isinstance(entities, dict):
        candidates = entities.values()
    else:
        candidates = entities

    normalized = []
    for entity in candidates:
        if isinstance(entity, EntityInfo):
            normalized.append(entity)
        elif isinstance(entity, dict):
            entity_info = _entity_info_from_dict(entity)
            if entity_info is not None:
                normalized.append(entity_info)
    return normalized


def _align_pairwise(
    entities1: List[EntityInfo],
    entities2: List[EntityInfo],
    model,
    pair_workers: int = 1,
    max_pairs_per_entity: Optional[int] = None,
    use_llm: bool = True,
) -> List[Tuple[EntityInfo, EntityInfo]]:

    aligned_pairs = []

    entities1 = _normalize_entities(entities1)
    entities2 = _normalize_entities(entities2)

    if not entities1 or not entities2:
        return aligned_pairs

    aligned_ent2_ids = set()


    pair_workers = max(1, pair_workers or 1)

    for ent1 in entities1:
        candidates = [ent2 for ent2 in entities2 if ent2.ent_id not in aligned_ent2_ids]
        if max_pairs_per_entity and max_pairs_per_entity > 0:
            candidates = candidates[:max_pairs_per_entity]

        exact_match = next((ent2 for ent2 in candidates if _is_exact_name_alignment(ent1, ent2)), None)
        if exact_match is not None:
            print(f"  Found exact alignment: ({ent1}) <-> ({exact_match})", flush=True)
            aligned_pairs.append((ent1, exact_match))
            aligned_ent2_ids.add(exact_match.ent_id)
            continue

        if not use_llm:
            continue

        if pair_workers == 1 or len(candidates) <= 1:
            for ent2 in candidates:
                if check_alignment(ent1, ent2, model):
                    print(f"  Found alignment: ({ent1}) <-> ({ent2})", flush=True)
                    aligned_pairs.append((ent1, ent2))
                    aligned_ent2_ids.add(ent2.ent_id)
                    break
            continue

        results = {}
        workers = min(pair_workers, len(candidates))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(check_alignment, ent1, ent2, model): index
                for index, ent2 in enumerate(candidates)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception:
                    results[index] = False

        for index, ent2 in enumerate(candidates):
            if ent2.ent_id in aligned_ent2_ids:
                continue

            if results.get(index):
                print(f"  Found alignment: ({ent1}) <-> ({ent2})", flush=True)
                aligned_pairs.append((ent1, ent2))
                aligned_ent2_ids.add(ent2.ent_id)
                break 
                
    return aligned_pairs


def align_entity_graphs_pairwise(
    umls_entities: List[EntityInfo],
    primekg_entities: List[EntityInfo],
    hetionet_entities: List[EntityInfo],
    model,
    pair_workers: int = 1,
    max_pairs_per_entity: Optional[int] = None,
    use_llm: bool = True,
) -> List[Tuple[EntityInfo, EntityInfo]]:
    all_aligned_pairs = []

    umls_primekg_pairs = _align_pairwise(
        umls_entities,
        primekg_entities,
        model,
        pair_workers=pair_workers,
        max_pairs_per_entity=max_pairs_per_entity,
        use_llm=use_llm,
    )
    all_aligned_pairs.extend(umls_primekg_pairs)

    umls_hetionet_pairs = _align_pairwise(
        umls_entities,
        hetionet_entities,
        model,
        pair_workers=pair_workers,
        max_pairs_per_entity=max_pairs_per_entity,
        use_llm=use_llm,
    )
    all_aligned_pairs.extend(umls_hetionet_pairs)
    
    primekg_hetionet_pairs = _align_pairwise(
        primekg_entities,
        hetionet_entities,
        model,
        pair_workers=pair_workers,
        max_pairs_per_entity=max_pairs_per_entity,
        use_llm=use_llm,
    )
    all_aligned_pairs.extend(primekg_hetionet_pairs)
    
    return all_aligned_pairs
