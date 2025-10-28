from typing import List, Dict, Set, Tuple, Optional, Any, Generator
import networkx as nx
from dataclasses import dataclass

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


class GraphFuser:
    def __init__(self, chains_per_kg: Dict[str, List[List[Tuple]]],
                aligned_entities: List[Tuple[EntityInfo, EntityInfo]]):
        self.chains_per_kg = chains_per_kg
        self.aligned_entities = aligned_entities
        self.graph = nx.MultiDiGraph() 
        self._merge_map = self._create_merge_map()
        
        self._build_fused_graph()
    def _create_merge_map(self) -> Dict[Tuple[str, str], Tuple[str, str]]:
        parent = {}

        def find(item):
            if item not in parent:
                parent[item] = item
            if parent[item] == item:
                return item
            parent[item] = find(parent[item])
            return parent[item]

        def union(item1, item2):
            root1 = find(item1)
            root2 = find(item2)
            if root1 != root2:
                parent[root1] = root2

        for ent1, ent2 in self.aligned_entities:
            id1 = ent1.get_tuple_id()
            id2 = ent2.get_tuple_id()
            union(id1, id2)

        all_entities = set(parent.keys())
        merge_map = {entity: find(entity) for entity in all_entities}
        
        return merge_map

    def _get_representative(self, source: str, entity_name: str) -> Tuple[str, str]:

        original_id = (source, entity_name)
        return self._merge_map.get(original_id, original_id)

    def _build_fused_graph(self):

        for source, chains in self.chains_per_kg.items():
            for path in chains:
                for triple in path:

                    if len(triple) < 3:
                        continue
                    head, relation, tail = triple[0], triple[1], triple[2]
                    score = triple[3] if len(triple) > 3 else 1.0


                    head_repr = self._get_representative(source, head)
                    tail_repr = self._get_representative(source, tail)


                    self.graph.add_edge(
                        head_repr,
                        tail_repr,
                        key=relation, 
                        relation=relation,
                        score=score,
                        original_source=source
                    )

    def _find_all_paths_dfs(self, start_node: Tuple[str, str]) -> Generator[List[Tuple], None, None]:
        stack = [(start_node, [], {start_node})]

        while stack:
            current_node, path_edges, path_nodes_set = stack.pop()

            successors = list(self.graph.successors(current_node))
            if not successors:
                if path_edges:  
                    yield path_edges
                continue


            for neighbor in reversed(successors):
                if neighbor in path_nodes_set:
                    continue

                edge_data_dict = self.graph.get_edge_data(current_node, neighbor)
                for relation in reversed(list(edge_data_dict.keys())):
                    new_edge = (current_node, relation, neighbor)
                    new_path_edges = path_edges + [new_edge]

                    new_path_nodes_set = path_nodes_set.copy()
                    new_path_nodes_set.add(neighbor)
                    
                    stack.append((neighbor, new_path_edges, new_path_nodes_set))

    def traverse_from_node_dfs(self, start_node_id: Tuple[str, str]) -> List[List[Tuple]]:

        if not self.graph.has_node(start_node_id):
            return []

        paths = list(self._find_all_paths_dfs(start_node_id))
        
        return paths

    def traverse_from_zero_in_degree_dfs(self) -> List[List[Tuple]]:

        start_nodes = [node for node, degree in self.graph.in_degree() if degree == 0]
        
        if not start_nodes:
            return []

        
        all_paths = []
        for start_node in start_nodes:
            paths_from_node = list(self._find_all_paths_dfs(start_node))
            if paths_from_node:
                all_paths.extend(paths_from_node)

        return all_paths
    
    def get_all_candidate_paths(self, custom_start_nodes: List[Tuple[str, str]]) -> List[List[Tuple]]:
        zero_degree_nodes = {node for node, degree in self.graph.in_degree() if degree == 0}
        
        all_start_nodes = zero_degree_nodes.union(set(custom_start_nodes))



        all_paths = []
        seen_paths = set()  

        for start_node in all_start_nodes:

            if not self.graph.has_node(start_node):
                continue

            paths_from_node = self._find_all_paths_dfs(start_node)
            for path in paths_from_node:
                path_tuple = tuple(path)
                if path_tuple not in seen_paths:
                    all_paths.append(path)
                    seen_paths.add(path_tuple)
        return all_paths

    def generate_path_descriptions(self, paths: List[List[Tuple]], model: Any) -> List[Tuple[List[Tuple], str]]:
        paths_with_descriptions = []
        for i, path in enumerate(paths):
            prompt_template = """
            You are a biomedical expert specializing in explaining complex relationships from knowledge graphs.
            Your task is to convert the following reasoning chain from a medical knowledge graph into a coherent, easy-to-understand natural language sentence or paragraph.

            The path is represented as a sequence of (Head Entity, Relation, Tail Entity) triples.
            Focus on creating a fluid narrative. Start from the first entity and explain how it connects to the last. Explain the meaning of the relations in a natural way. Do not just list the triples.

            EXAMPLE:
            Reasoning Chain: [(('umls', 'Migraine Disorders'), 'has_symptom', ('umls', 'Headache')), (('umls', 'Headache'), 'may_be_treated_by', ('umls', 'Paracetamol'))]
            Natural Language Explanation: Migraine disorders are characterized by the symptom of a headache, which in turn may be treated by Paracetamol.

            Now, generate the explanation for the following chain.

            Reasoning Chain: {}
            Natural Language Explanation:
            """
            prompt = prompt_template.format(path)
            
            description = model.generate_response(prompt, 256, 0.4)
            paths_with_descriptions.append((path, description))

        return paths_with_descriptions

    def filter_paths_by_relevance(self, query: str, paths_with_descriptions: List[Tuple[List[Tuple], str]], model: Any) -> List[Tuple[List[Tuple], str]]:
        relevant_paths = []
        prompt_template = """
        Given the following medical question, a reasoning chain retrieved from a knowledge graph, and a natural language explanation description from the reasoning chain, determine whether the explanation is relevant to the question.

        Question: {}
        Reasoning Chain: {}
        Natural Language Description: {}

        Task: Analyze the relevance of the description to the question. Respond with "Yes" if the description is relevant, or "No" if the description is not relevant.
        """
        for i, (path, description) in enumerate(paths_with_descriptions):
            prompt = prompt_template.format(query, path, description)
            
            response = model.generate_response(prompt, 256, 0.0) #
        
            if response.strip().lower() == 'yes':
                relevant_paths.append((path, description))
            else:
                pass
        return relevant_paths