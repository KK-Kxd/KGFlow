from neo4j import GraphDatabase


class HetionetDatabase:
    def __init__(self, uri, user, password, name):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.name = name
    
    def close(self):
        self.driver.close()

    def get_name(self):
        return self.name

    @staticmethod
    def _escape_relationship(relationship):
        return relationship.replace("`", "``")

    def resolve_entity(self, entity):
        if isinstance(entity, dict):
            identifier = entity.get("identifier") or entity.get("id") or entity.get("ent_id")
            name = entity.get("name")
            if identifier:
                concept = self.get_concept_by_identifier(identifier)
                if concept:
                    return {
                        "id": concept["identifier"],
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

        concept = self.get_concept_by_identifier(value) or self.get_concept_by_name(value)
        if not concept:
            return None

        return {
            "id": concept["identifier"],
            "name": concept["name"],
            "type": concept.get("type", "unknown"),
            "source": self.name,
        }

    def get_concept_by_name(self, entity_name):
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n {name: $entity_name})
                RETURN n.identifier AS identifier,
                       n.name AS name,
                       n.source AS source,
                       labels(n)[0] AS type
                LIMIT 1
                """,
                entity_name=entity_name
            )
            record = result.single()
            return record.data() if record else None
        
    def get_concept_by_identifier(self, entity_identifier):
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n)
                WHERE n.identifier = $entity_identifier
                   OR toString(n.identifier) = toString($entity_identifier)
                RETURN n.identifier AS identifier,
                       n.name AS name,
                       n.source AS source,
                       n.license AS license,
                       n.url AS url,
                       labels(n)[0] AS type
                LIMIT 1
                """,
                entity_identifier=entity_identifier
            )
            record = result.single()
            return record.data() if record else None

    def get_entity_relationships(self, entity_identifier):
        concept = self.resolve_entity(entity_identifier)
        if not concept:
            return [], []
        entity_identifier = concept["id"]

        with self.driver.session() as session:
            outgoing_result = session.run(
                """
                MATCH (n {identifier: $entity_identifier})-[r]->(m)
                RETURN DISTINCT type(r) AS relationship
                """,
                entity_identifier=entity_identifier
            )
            incoming_result = session.run(
                """
                MATCH (m)-[r]->(n {identifier: $entity_identifier})
                RETURN DISTINCT type(r) AS relationship
                """,
                entity_identifier=entity_identifier
            )
            
            outgoing_relationships = [record["relationship"] for record in outgoing_result]
            incoming_relationships = [record["relationship"] for record in incoming_result]

            return outgoing_relationships or [], incoming_relationships or []
        
    def get_entity_relationships_by_name(self, entity_name):
        return self.get_entity_relationships(entity_name)

    def find_tail_concepts(self, head_entity_identifier, relationship):
        concept = self.resolve_entity(head_entity_identifier)
        if not concept:
            return []
        head_entity_identifier = concept["id"]
        relationship = self._escape_relationship(relationship)

        with self.driver.session() as session:
            result = session.run(
                f"MATCH (n {{identifier: $head_entity_identifier}})-[:`{relationship}`]->(m) " +
                "RETURN m.name AS tail_entity",
                head_entity_identifier=head_entity_identifier
            )
            return [record["tail_entity"] for record in result]
            
    def find_head_concepts(self, tail_entity_identifier, relationship):
        """查找给定尾实体和关系的头实体"""
        concept = self.resolve_entity(tail_entity_identifier)
        if not concept:
            return []
        tail_entity_identifier = concept["id"]
        relationship = self._escape_relationship(relationship)

        with self.driver.session() as session:
            result = session.run(
                f"MATCH (m)-[:`{relationship}`]->(n {{identifier: $tail_entity_identifier}}) " +
                "RETURN m.name AS head_entity",
                tail_entity_identifier=tail_entity_identifier
            )
            return [record["head_entity"] for record in result]

    def find_tail_entity(self, head_entity_identifier, relationship):
        return self.find_tail_concepts(head_entity_identifier, relationship)

    def find_head_entity(self, tail_entity_identifier, relationship):
        return self.find_head_concepts(tail_entity_identifier, relationship)


    def get_neighbors(self, node_name, outgoing_limit=3, incoming_limit=3):

        concept_info = self.resolve_entity(node_name)
        if not concept_info:
            return []
        node_identifier = concept_info["id"]
        node_name = concept_info["name"]

        with self.driver.session() as session:

            outgoing_result = session.run(
                """
                MATCH (n {identifier: $node_identifier})-[r]->(neighbor)
                RETURN 
                    neighbor.identifier AS neighbor_cui, 
                    neighbor.name AS neighbor_name,
                    type(r) AS relation,
                    'outgoing' AS direction
                LIMIT $limit
                """,
                node_identifier=node_identifier, limit=outgoing_limit
            )
            
 
            incoming_result = session.run(
                """
                MATCH (n {identifier: $node_identifier})<-[r]-(neighbor)
                RETURN 
                    neighbor.identifier AS neighbor_cui, 
                    neighbor.name AS neighbor_name,
                    type(r) AS relation,
                    'incoming' AS direction
                LIMIT $limit
                """,
                node_identifier=node_identifier, limit=incoming_limit
            )

            neighbors = []

            for record in outgoing_result:
                neighbor_name = record["neighbor_name"] or ""
                neighbors.append([node_name, record["relation"], neighbor_name])
            

            for record in incoming_result:
                neighbor_name = record["neighbor_name"] or ""
                neighbors.append([neighbor_name, record["relation"], node_name])
                
            return neighbors
