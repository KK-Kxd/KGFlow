from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from typing import List, Tuple, Optional, Dict

class UMLSDatabase:
    def __init__(self, uri: str, user: str, password: str, name: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._create_constraints()
        self.name = name

    def get_name(self):
        return self.name

    def close(self) -> None:
        self.driver.close()

    @staticmethod
    def _preferred_name(names: List[str]) -> str:
        return min(names, key=len) if names else ""

    @staticmethod
    def _escape_relationship(relationship: str) -> str:
        return relationship.replace("`", "``")

    @staticmethod
    def _escape_fulltext_query(value: str) -> str:
        special_chars = set('+-!(){}[]^"~*?:\\/&|')
        return "".join(
            f"\\{char}" if char in special_chars else char
            for char in value
        )

    @staticmethod
    def _is_fulltext_query_error(exc: Exception) -> bool:
        message = str(exc)
        return (
            "TooManyClauses" in message
            or "ParseException" in message
            or "db.index.fulltext.queryNodes" in message
        )

    def _concept_record(self, cui: str, names: List[str]) -> Dict:
        return {
            "CUI": cui,
            "id": cui,
            "name": self._preferred_name(names or []),
            "names": names or [],
            "type": "Concept",
            "source": self.name,
        }

    def resolve_entity(self, entity) -> Optional[Dict]:
        if isinstance(entity, dict):
            cui = entity.get("CUI") or entity.get("cui") or entity.get("id") or entity.get("ent_id")
            name = entity.get("name")
            if cui:
                concept = self.get_concept_by_cui(str(cui))
                if concept:
                    return {
                        "id": concept["CUI"],
                        "name": concept["name"],
                        "type": "Concept",
                        "source": self.name,
                    }
            if name:
                return self.resolve_entity(name)
            return None

        value = str(entity).strip()
        if not value:
            return None

        concept = self.get_concept_by_cui(value) or self.get_concept_by_name(value)
        if not concept:
            return None

        return {
            "id": concept["CUI"],
            "name": concept.get("name") or self._preferred_name(concept.get("names", [])),
            "type": "Concept",
            "source": self.name,
        }

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
            try:
                result = session.run("""
                    CALL db.index.fulltext.queryNodes("concept_names", $name)
                    YIELD node
                    RETURN collect(node.CUI) AS cuis
                """, name=self._escape_fulltext_query(name))
                record = result.single()
            except Neo4jError as exc:
                if self._is_fulltext_query_error(exc):
                    return False, []
                raise

            cuis = record["cuis"] if record else []
            return len(cuis) > 0, cuis

    def get_entity_relationships(self, cui: str) -> Tuple[List[str], List[str]]:
        concept = self.resolve_entity(cui)
        if not concept:
            return [], []
        cui = concept["id"]

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
            if not record:
                return [], []
            outgoing = [rel for rel in (record["outgoing"] or []) if rel]
            incoming = [rel for rel in (record["incoming"] or []) if rel]
            return outgoing, incoming


    def get_concept_by_name(self, name: str) -> dict:

        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Concept)
                WHERE $name IN c.names
                RETURN c.CUI AS CUI, c.names AS names
                LIMIT 1
            """, name=name)
            record = result.single()
            if record:
                data = record.data()
                data["name"] = self._preferred_name(data.get("names", []))
                return data

            try:
                result = session.run("""
                    CALL db.index.fulltext.queryNodes("concept_names", $name)
                    YIELD node, score
                    RETURN node.CUI AS CUI, node.names AS names
                    ORDER BY score DESC
                    LIMIT 1
                """, name=self._escape_fulltext_query(name))
                record = result.single()
            except Neo4jError as exc:
                if self._is_fulltext_query_error(exc):
                    return None
                raise
            if not record:
                return None
            data = record.data()
            data["name"] = self._preferred_name(data.get("names", []))
            return data

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
            preferred_name = self._preferred_name(names)

            return {
                "CUI": data["CUI"],
                "names": names,
                "name": preferred_name
            }

    def find_tail_concepts(self, source_cui: str, relation_type: str) -> List[str]:
        concept = self.resolve_entity(source_cui)
        if not concept:
            return []
        source_cui = concept["id"]
        relation_type = self._escape_relationship(relation_type)

        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (:Concept {{CUI: $source_cui}})-[:`{relation_type}`]->(target:Concept)
                RETURN target.CUI AS CUI, target.names AS names
                """,
                source_cui=source_cui
            )
            return [self._concept_record(record["CUI"], record["names"] or []) for record in result]

    def find_head_concepts(self, cui: str, relation_type: str) -> List[str]:
        concept = self.resolve_entity(cui)
        if not concept:
            return []
        cui = concept["id"]
        relation_type = self._escape_relationship(relation_type)

        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (target:Concept {{CUI: $cui}})<-[:`{relation_type}`]-(source:Concept)
                RETURN source.CUI AS CUI, source.names AS names
                """,
                cui=cui
            )
            return [self._concept_record(record["CUI"], record["names"] or []) for record in result]

    def find_tail_entity(self, source_cui: str, relation_type: str) -> List[str]:
        return self.find_tail_concepts(source_cui, relation_type)

    def find_head_entity(self, cui: str, relation_type: str) -> List[str]:
        return self.find_head_concepts(cui, relation_type)

        
 

    def get_neighbors(self, cui: str, outgoing_limit: int = 3, incoming_limit: int = 3) -> List[List[str]]:
        concept = self.resolve_entity(cui)
        if not concept:
            return []

        cui = concept["id"]
        name = concept["name"]

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

                shortest_name = self._preferred_name(names)
                
                neighbors.append([name,record["relation"],shortest_name])

            

            for record in incoming_result:
                names = record["neighbor_names"] or [""]

                shortest_name = self._preferred_name(names)
                
                neighbors.append([shortest_name,record["relation"],name])

                
            return neighbors


    


