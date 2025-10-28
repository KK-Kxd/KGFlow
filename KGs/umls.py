from neo4j import GraphDatabase
from typing import List, Tuple, Optional, Dict
import re

from neo4j import GraphDatabase
from typing import List, Tuple, Dict

class UMLSDatabase:
    def __init__(self, uri: str, user: str, password: str, name: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._create_constraints()
        self.name = name

    def get_name(self):
        return self.name

    def close(self) -> None:
        self.driver.close()

    def _create_constraints(self) -> None:

        with self.driver.session() as session:
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS 
                FOR (c:Concept) REQUIRE c.CUI IS UNIQUE
            """)
            session.run("""
                CREATE FULLTEXT INDEX concept_names IF NOT EXISTS 
                FOR (c:Concept) ON EACH [c.names]
            """)

    def entity_exists(self, name: str) -> Tuple[bool, List[str]]:

        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.fulltext.queryNodes("concept_names", $name)
                YIELD node
                RETURN collect(node.CUI) AS cuis
            """, name=name)
            
            record = result.single()
            cuis = record["cuis"] if record else []
            return len(cuis) > 0, cuis

    def get_entity_relationships(self, cui: str) -> Tuple[List[str], List[str]]:

        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Concept {CUI: $cui})
                OPTIONAL MATCH (c)-[r_out]->()
                WITH c, collect(DISTINCT CASE WHEN r_out IS NOT NULL THEN type(r_out) END) AS outgoing
                OPTIONAL MATCH (c)<-[r_in]-()
                RETURN 
                    outgoing,
                    collect(DISTINCT CASE WHEN r_in IS NOT NULL THEN type(r_in) END) AS incoming
            """, cui=cui)
            record = result.single()
            return record["outgoing"] or [], record["incoming"] or []


    def get_concept_by_name(self, name: str) -> dict:

        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Concept)
                WHERE $name IN c.names
                RETURN c.CUI AS CUI, c.names AS names
                LIMIT 1
            """, name=name)
            record = result.single()
            return record.data() if record else None

    def get_concept_by_cui(self, cui: str) -> Optional[Dict]:

        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Concept {CUI: $cui})
                RETURN c.CUI AS CUI, c.names AS names
            """, cui=cui)

            record = result.single()
            if not record:
                return None

            data = record.data()
            names = data.get("names", [])

            # 选择最短的名称作为首选名称
            preferred_name = min(names, key=len) if names else ""

            return {
                "CUI": data["CUI"],
                "names": names,
                "name": preferred_name
            }

    def find_tail_concepts(self, source_cui: str, relation_type: str) -> List[Dict]:

        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (:Concept {{CUI: $source_cui}})-[:`{relation_type}`]->(target:Concept)
                RETURN target.CUI AS CUI, target.names AS names
                """,
                source_cui=source_cui
            )
            return [min(record["names"], key=len) if record["names"] else "" for record in result]

    def find_head_concepts(self, cui: str, relation_type: str) -> List[Dict]:
        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (target:Concept {{CUI: $cui}})<-[:`{relation_type}`]-(source:Concept)
                RETURN source.names AS names
                """,
                cui=cui
            )
            return [min(record["names"], key=len) if record["names"] else "" for record in result]

        
 

    def get_neighbors(self, cui: str, outgoing_limit: int = 3, incoming_limit: int = 3) -> List[Dict]:
       
        name = self.get_concept_by_cui(cui).get("name")

        with self.driver.session() as session:
            outgoing_result = session.run("""
                MATCH (c:Concept {CUI: $cui})-[r]->(neighbor:Concept)
                RETURN 
                    neighbor.CUI AS neighbor_cui, 
                    neighbor.names AS neighbor_names,
                    type(r) AS relation,
                    'outgoing' AS direction
                LIMIT $limit
            """, cui=cui, limit=outgoing_limit)
            
            incoming_result = session.run("""
                MATCH (c:Concept {CUI: $cui})<-[r]-(neighbor:Concept)
                RETURN 
                    neighbor.CUI AS neighbor_cui, 
                    neighbor.names AS neighbor_names,
                    type(r) AS relation,
                    'incoming' AS direction
                LIMIT $limit
            """, cui=cui, limit=incoming_limit)
            

            neighbors = []
            

            for record in outgoing_result:
                names = record["neighbor_names"] or [""]

                shortest_name = min(names, key=len) if names else ""
                
                neighbors.append([name,record["relation"],shortest_name])

            

            for record in incoming_result:
                names = record["neighbor_names"] or [""]

                shortest_name = min(names, key=len) if names else ""
                
                neighbors.append([shortest_name,record["relation"],name])

                
            return neighbors


    



if __name__ == '__main__':
    db = UMLSDatabase("bolt://localhost:7689", "neo4j", "password", "UMLS")
    # print(db.get_concept_by_cui("C5238205"))
    # _,cuis = db.entity_exists("migraine")
    # print(len(cuis))
    # concept = db.get_concept_by_name("Dupilumab")
    # print(type(concept))
    # print(concept)
    # neighbors = db.get_neighbors("C0149931",10,10)
    # print(neighbors)
    # # out_rel, in_rel = db.get_entity_relationships("C0000741")
    # # print(type(out_rel), in_rel)
    # tail = db.find_tail_concepts(concept["CUI"],'may_be_treated_by')
    # print(tail)
    # print(db.find_head_concepts(concept["CUI"],'may_be_treated_by'))

    paths = db.get_path_between_concepts("SATB1", "thymocytes")
    print(paths)
    # print(db.get_all_paths_between_concepts("pancreatic juice", "pancreatic diseases"))

