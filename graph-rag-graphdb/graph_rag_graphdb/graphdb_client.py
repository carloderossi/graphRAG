from neo4j import GraphDatabase
import numpy as np
import ollama

EMBED_MODEL = "mxbai-embed-large:latest"
LLM_MODEL = "llama3.1:8b"

# -----------------------------
# 1. Neo4j connection
# -----------------------------
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password123") # since it can only be accessed via localhost and public data only

driver = GraphDatabase.driver(URI, auth=AUTH)

# -----------------------------
# 2. Embed the question
# -----------------------------
def embed(text: str):
    res = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return res["embedding"]


# -----------------------------
# 3. Retrieve top‑k semantic chunks
# -----------------------------
def semantic_search(question_embedding, k=5):
    query = """
    MATCH (c:Chunk)
    RETURN c.id AS id, c.text AS text, c.embedding AS emb
    """
    with driver.session() as session:
        rows = session.run(query).data()

    scored = []
    q = np.array(question_embedding)

    for row in rows:
        emb = np.array(row["emb"])
        score = np.dot(q, emb) / (np.linalg.norm(q) * np.linalg.norm(emb))
        scored.append((score, row["id"], row["text"]))

    scored.sort(reverse=True)
    return scored[:k]

# -----------------------------
# 4. Expand into the KG
# -----------------------------
def expand_graph(chunk_ids):
    # query = """
    # MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
    # WHERE c.id IN $chunk_ids
    # OPTIONAL MATCH (e)-[r:RELATION]->(t:Entity)
    # RETURN c.id AS chunk_id, e.name AS entity, e.type AS type,
    #        type(r) AS rel_type, t.name AS target
    # """
    # query = """
    # MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
    # WHERE c.id IN $chunk_ids
    # AND e.type = "Provider"

    # MATCH (e)-[r:RELATION]->(t:Entity)
    # WHERE r.type STARTS WITH "MUST"

    # RETURN
    #     c.id AS chunk_id,
    #     e.name AS provider,
    #     r.type AS obligation,
    #     t.name AS target
    # ORDER BY chunk_id, obligation;
    # """
    query = """
    MATCH (c:Chunk)-[:MENTIONS]->(prov:Entity {type: "Provider"})
    WHERE c.id IN $chunk_ids

    MATCH (prov)-[r:RELATION]->(t:Entity)
    WHERE r.type STARTS WITH "MUST"

    RETURN
        prov.name AS provider,
        r.type AS obligation,
        t.name AS target
    ORDER BY obligation;
    """
    with driver.session() as session:
        return session.run(query, chunk_ids=chunk_ids).data()

# -----------------------------
# 5. Ask a question
# -----------------------------
def ask(question):
    print(f"\n🔍 Question: {question}")

    q_emb = embed(question)
    top_chunks = semantic_search(q_emb, k=5)

    print("\nTop semantic chunks:")
    for score, cid, text in top_chunks:
        print(f"  • {cid}  (score={score:.3f})")

    chunk_ids = [cid for _, cid, _ in top_chunks]
    kg = expand_graph(chunk_ids)

    print("\nGraph expansion results:")
    for row in kg:
        print(row)

    return kg

# -----------------------------
# Run the query
# -----------------------------
if __name__ == "__main__":
    ask("What obligations does the EU AI Act impose on Providers?")