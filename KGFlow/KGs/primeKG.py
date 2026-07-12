from neo4j import GraphDatabase



class PrimeDatabase:
    def __init__(self, uri, user, password, name, database=None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.name = name
        self.database = database

    def _session(self):
        if self.database:
            return self.driver.session(database=self.database)
        return self.driver.session()

    def close(self):
        self.driver.close()

    def get_name(self):
        return self.name

    @staticmethod
    def _escape_relationship(relationship):
        return relationship.replace("`", "``")

    def resolve_entity(self, entity):
        if isinstance(entity, dict):
            entity_id = (
                entity.get("node_index")
                or entity.get("index")
                or entity.get("CUI")
                or entity.get("cui")
                or entity.get("id")
                or entity.get("ent_id")
            )
            name = entity.get("node_name") or entity.get("name")
            if entity_id:
                concept = self.get_concept_by_cui(str(entity_id))
                if concept:
                    return {
                        "id": concept["index"],
                        "name": concept["name"],
                        "type": concept.get("type", "unknown"),
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
            "id": concept["index"],
            "name": concept["name"],
            "type": concept.get("type", "unknown"),
            "source": self.name,
        }

    def get_concept_by_name(self, entity_name):
        with self._session() as session:
            result = session.run(
                """
                MATCH (n {node_name: $entity_name})
                RETURN n.node_index AS CUI,
                       n.node_name AS name,
                       n.node_index AS index,
                       labels(n)[0] AS type
                LIMIT 1
                """,
                entity_name=entity_name
            )
            record = result.single()
            return record.data() if record else None
        
    def get_concept_by_cui(self, entity_id):
        with self._session() as session:
            result = session.run(
                """
                MATCH (n)
                WHERE n.node_index = $entity_id
                   OR toString(n.node_index) = toString($entity_id)
                   OR n.node_id = $entity_id
                RETURN n.node_id AS CUI,
                       n.node_name AS name,
                       n.node_index AS index,
                       labels(n)[0] AS type
                LIMIT 1
                """,
                entity_id=entity_id
            )
            
            record = result.single()
            return record.data() if record else None


    def get_entity_relationships(self, entity_id):
        concept = self.resolve_entity(entity_id)
        if not concept:
            return [], []
        entity_id = concept["id"]

        with self._session() as session:
            outgoing_result = session.run(
                """
                MATCH (n {node_index: $entity_id})-[r]->(m)
                RETURN DISTINCT type(r) AS relationship
                """,
                entity_id=entity_id
            )
            incoming_result = session.run(
                """
                MATCH (m)-[r]->(n {node_index: $entity_id})
                RETURN DISTINCT type(r) AS relationship
                """,
                entity_id=entity_id
            )
            
            outgoing_relationships = [record["relationship"] for record in outgoing_result]
            incoming_relationships = [record["relationship"] for record in incoming_result]

            return outgoing_relationships or [], incoming_relationships or []
        
    def get_entity_relationships_name(self, entity_id):
        return self.get_entity_relationships(entity_id)

    def find_tail_concepts(self, head_entity_id, relationship):
        concept = self.resolve_entity(head_entity_id)
        if not concept:
            return []
        head_entity_id = concept["id"]
        relationship = self._escape_relationship(relationship)

        with self._session() as session:
            result = session.run(
                f"MATCH (n {{node_index: $head_entity_id}})-[:`{relationship}`]->(m) RETURN m.node_name AS tail_entity",
                head_entity_id=head_entity_id
            )
            return [record["tail_entity"] for record in result]
 
    def find_head_concepts(self, tail_entity_id, relationship):
        concept = self.resolve_entity(tail_entity_id)
        if not concept:
            return []
        tail_entity_id = concept["id"]
        relationship = self._escape_relationship(relationship)

        with self._session() as session:
            result = session.run(
                f"MATCH (m)-[:`{relationship}`]->(n {{node_index: $tail_entity_id}}) RETURN m.node_name AS head_entity",
                tail_entity_id=tail_entity_id
            )
            return [record["head_entity"] for record in result]

    def find_tail_entity(self, head_entity_id, relationship):
        return self.find_tail_concepts(head_entity_id, relationship)

    def find_head_entity(self, tail_entity_id, relationship):
        return self.find_head_concepts(tail_entity_id, relationship)

    def get_neighbors(self, node_name, outgoing_limit=3, incoming_limit=3):


        concept_info = self.resolve_entity(node_name)
        if not concept_info:
            return []
        node_index = concept_info["id"]
        node_name = concept_info["name"]

        with self._session() as session:

            outgoing_result = session.run(
                """
                MATCH (n {node_index: $node_index})-[r]->(neighbor)
                RETURN 
                    neighbor.node_index AS neighbor_cui, 
                    neighbor.node_name AS neighbor_name,
                    type(r) AS relation,
                    'outgoing' AS direction
                LIMIT $limit
                """,
                node_index=node_index, limit=outgoing_limit
            )

            incoming_result = session.run(
                """
                MATCH (n {node_index: $node_index})<-[r]-(neighbor)
                RETURN 
                    neighbor.node_index AS neighbor_cui, 
                    neighbor.node_name AS neighbor_name,
                    type(r) AS relation,
                    'incoming' AS direction
                LIMIT $limit
                """,
                node_index=node_index, limit=incoming_limit
            )
            

            neighbors = []
            

            for record in outgoing_result:
                neighbor_name = record["neighbor_name"] or ""
                neighbors.append([node_name,record["relation"],neighbor_name])
            

            for record in incoming_result:
                neighbor_name = record["neighbor_name"] or ""
                neighbors.append([neighbor_name,record["relation"],node_name])
                
            return neighbors
