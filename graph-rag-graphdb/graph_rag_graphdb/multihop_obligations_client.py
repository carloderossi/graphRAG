from neo4j import GraphDatabase
import numpy as np
import ollama

EMBED_MODEL = "mxbai-embed-large:latest"
LLM_MODEL = "llama3.1:8b"

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password123")

driver = GraphDatabase.driver(URI, auth=AUTH)


# ---------------------------------------------------------
# 1. Embedding
# ---------------------------------------------------------
def embed(text: str):
    res = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return res["embedding"]


# ---------------------------------------------------------
# 2. Semantic Search (Python-side cosine)
# ---------------------------------------------------------
def semantic_search(question_embedding, k=5):
    query = """
    MATCH (c:Chunk)
    RETURN c.id AS id, c.text AS text, c.embedding AS emb
    """
    with driver.session() as session:
        rows = session.run(query).data()

    q = np.array(question_embedding)
    scored = []

    for row in rows:
        emb = np.array(row["emb"])
        score = np.dot(q, emb) / (np.linalg.norm(q) * np.linalg.norm(emb))
        scored.append((score, row["id"], row["text"]))

    scored.sort(reverse=True)
    return scored[:k]


# ---------------------------------------------------------
# 3. Obligation Queries
# ---------------------------------------------------------

# A. Direct obligations involving Provider (both directions)
QUERY_DIRECT = """
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
WHERE c.id IN $chunk_ids AND e.name = "Provider"

MATCH (e)-[r:RELATION {type: "OBLIGATION"}]-(other:Entity)
RETURN 
    e.name AS provider,
    r.type AS obligation_type,
    other.name AS related_entity,
    c.id AS source_chunk
ORDER BY related_entity;
"""

# B. Obligations in same chunk as Provider
QUERY_SAME_CHUNK = """
MATCH (c:Chunk)
WHERE c.id IN $chunk_ids

MATCH (c)-[:MENTIONS]->(prov:Entity)
WHERE prov.name = "Provider"

MATCH (c)-[:MENTIONS]->(actor:Entity)
MATCH (actor)-[r:RELATION {type: "OBLIGATION"}]->(target:Entity)
RETURN
    actor.name AS actor,
    r.type AS obligation_type,
    target.name AS target,
    c.id AS source_chunk
ORDER BY actor, target;
"""

# C. All obligations in top-k chunks (regardless of actor)
QUERY_ALL_OBLIGATIONS = """
MATCH (c:Chunk)
WHERE c.id IN $chunk_ids

MATCH (c)-[:MENTIONS]->(actor:Entity)
MATCH (actor)-[r:RELATION {type: "OBLIGATION"}]->(target:Entity)
RETURN
    actor.name AS actor,
    r.type AS obligation_type,
    target.name AS target,
    c.id AS source_chunk
ORDER BY actor, target;
"""

# D. Multi-hop obligation reasoning (1–2 hops)
QUERY_MULTI_HOP = """
MATCH (c:Chunk)
WHERE c.id IN $chunk_ids

MATCH (c)-[:MENTIONS]->(prov:Entity)
WHERE prov.name = "Provider"

MATCH path = (prov)-[:RELATION*1..2]->(x)
WHERE ANY(rel IN relationships(path) WHERE rel.type = "OBLIGATION")
RETURN path;
"""


# ---------------------------------------------------------
# 4. Run all queries
# ---------------------------------------------------------
def expand_obligations(chunk_ids):
    with driver.session() as session:
        direct = session.run(QUERY_DIRECT, chunk_ids=chunk_ids).data()
        same_chunk = session.run(QUERY_SAME_CHUNK, chunk_ids=chunk_ids).data()
        all_obl = session.run(QUERY_ALL_OBLIGATIONS, chunk_ids=chunk_ids).data()
        multi_hop = session.run(QUERY_MULTI_HOP, chunk_ids=chunk_ids).data()

    return {
        "direct_obligations": direct,
        "same_chunk_obligations": same_chunk,
        "all_obligations_in_chunks": all_obl,
        "multi_hop_obligations": multi_hop,
    }


# ---------------------------------------------------------
# 5. Ask a question
# ---------------------------------------------------------
def ask(question):
    print(f"\n🔍 Question: {question}")

    q_emb = embed(question)
    top_chunks = semantic_search(q_emb, k=5)

    print("\nTop semantic chunks:")
    for score, cid, text in top_chunks:
        print(f"  • {cid}  (score={score:.3f})")

    chunk_ids = [cid for _, cid, _ in top_chunks]

    results = expand_obligations(chunk_ids)

    print("\nGraph expansion results:")
    for key, value in results.items():
        print(f"\n--- {key} ---")
        for row in value:
            print(row)

    return results


# ---------------------------------------------------------
# 6. Run
# ---------------------------------------------------------
if __name__ == "__main__":
    ask("What obligations does the EU AI Act impose on Providers?")