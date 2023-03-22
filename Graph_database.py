from neo4j import GraphDatabase

# Set the named entities
entities = [
    "T-cell",
    "cytokines",
    "transcription factor",
]

# Set the sentences to be used for training
sentences = [
    "T-cells produce cytokines",
    "Cytokines activate transcription factors",
    "Transcription factors regulate T-cell function",
]

# Set the relations and their types
relations = [
    ("T-cell", "cytokines", "produce"),
    ("cytokines", "transcription factor", "activate"),
    ("transcription factor", "T-cell", "regulate"),
]
relation_types = [
    "production",
    "activation",
    "regulation",
]

# Define the Neo4j driver and session
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
session = driver.session()

# Create the named entity nodes
for entity in entities:
    session.run("MERGE (n:Entity {name: $name})", name=entity)

# Create the sentence nodes and connect them to the named entity nodes
for sentence in sentences:
    parts = sentence.split()
    entity1 = parts[0]
    entity2 = parts[2]
    relation_type = relations[entities.index(entity1)][2]
    session.run("""
        MATCH (e1:Entity {name: $entity1})
        MATCH (e2:Entity {name: $entity2})
        MERGE (s:Sentence {text: $text})
        MERGE (e1)-[:%s]->(e2)
        MERGE (s)-[:mentions]->(e1)
        MERGE (s)-[:mentions]->(e2)
    """ % relation_type, entity1=entity1, entity2=entity2, text=sentence)

# Create the relationship type nodes and connect them to the relationship edges
for relation_type in relation_types:
    session.run("MERGE (n:RelationType {name: $name})", name=relation_type)
    for relation in relations:
        entity1 = relation[0]
        entity2 = relation[1]
        rel_type = relation[2]
        if rel_type == relation_type:
            session.run("""
                MATCH (e1:Entity {name: $entity1})-[r]->(e2:Entity {name: $entity2})
                MATCH (t:RelationType {name: $type})
                MERGE (r)-[:ofType]->(t)
            """, entity1=entity1, entity2=entity2, type=relation_type)

# Close the session and driver
session.close()
driver.close()