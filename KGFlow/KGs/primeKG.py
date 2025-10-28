from neo4j import GraphDatabase



class PrimeDatabase:
    def __init__(self, uri, user, password, name):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.name = name
        self.database = name
    def close(self):
        self.driver.close()

    def get_name(self):
        return self.name

    def get_concept_by_name(self, entity_name):
        with self.driver.session(database=self.database) as session:
            result = session.run(
                "MATCH (n {node_name: $entity_name}) RETURN n.node_index AS CUI, n.node_name AS name", entity_name=entity_name
            )
            record = result.single()
            return record.data() if record else None
        
    def get_concept_by_cui(self, entity_id):
        with self.driver.session(database=self.database) as session:
            result = session.run(
                "MATCH (n {node_index: $entity_id}) RETURN " \
                "n.node_id AS CUI, n.node_name AS name, n.node_index AS index", entity_id=entity_id
            )
            
            record = result.single()
            return record.data() if record else None


    def get_entity_relationships(self, entity_id):
        with self.driver.session(database=self.database) as session:
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
        with self.driver.session(database=self.database) as session:
            outgoing_result = session.run(
                """
                MATCH (n {node_name: $entity_id})-[r]->(m)
                RETURN DISTINCT type(r) AS relationship
                """,
                entity_id=entity_id
            )
            incoming_result = session.run(
                """
                MATCH (m)-[r]->(n {node_name: $entity_id})
                RETURN DISTINCT type(r) AS relationship
                """,
                entity_id=entity_id
            )
            
            outgoing_relationships = [record["relationship"] for record in outgoing_result]
            incoming_relationships = [record["relationship"] for record in incoming_result]

            return outgoing_relationships or [], incoming_relationships or []

    def find_tail_concepts(self, head_entity_name, relationship):
        with self.driver.session(database=self.database) as session:
            result = session.run(
                "MATCH (n {node_name: $head_entity_name})-[:`" + relationship + "`]->(m) RETURN m.node_name AS tail_entity",
                head_entity_name=head_entity_name
            )
        return [record["tail_entity"] for record in result]
 
    def find_head_concepts(self, tail_entity_name, relationship):
        with self.driver.session(database=self.database) as session:
            result = session.run(
                "MATCH (m)-[:`" + relationship + "`]->(n {node_name: $tail_entity_name}) RETURN m.node_name AS head_entity",
                tail_entity_name=tail_entity_name
            )
        return [record["head_entity"] for record in result]

    def get_neighbors(self, node_name, outgoing_limit=3, incoming_limit=3):


        concept_info = self.get_concept_by_name(node_name)
        node_index = concept_info.get("cui", "") if concept_info else ""

        with self.driver.session(database=self.database) as session:

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

