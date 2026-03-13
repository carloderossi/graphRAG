import json
import ollama
import chromadb
from utils.config import EMBED_MODEL, LLM_MODEL, SEMANTIC_INDEX_PATH, KG_PATH, get_chroma_db_path
from collections import defaultdict

# ------------------------------
# OLLAMA HELPERS
# ------------------------------

def embed(text: str):
    res = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return res["embedding"]

def llm(prompt: str, temperature=0):
    res = ollama.generate(
        model=LLM_MODEL,
        prompt=prompt,
        options={"temperature": temperature}
    )
    return res["response"]

# ------------------------------
# LOAD SEMANTIC INDEX + COMMUNITIES + KG
# ------------------------------

def load_semantic_index():
    with open(SEMANTIC_INDEX_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = []
    for chunk_id, obj in data["chunks"].items():
        chunks.append({
            "chunk_id": chunk_id,
            "text": obj["text"],
            "source": obj.get("source", ""),
            "embedding": obj["vec"]
        })

    communities = data.get("communities", [])
    return chunks, communities

def load_kg():
    by_chunk = {}
    with open(KG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            by_chunk[obj["chunk_id"]] = obj
    return by_chunk

chunks, communities = load_semantic_index()
kg_by_chunk = load_kg()

# ------------------------------
# COMMUNITY INDEX
# ------------------------------

community_by_id = {c["community_id"]: c for c in communities}

chunk_to_community = {}
for c in communities:
    for cid in c["member_ids"]:
        chunk_to_community[cid] = c["community_id"]

# ------------------------------
# LOAD CHROMA DB
# ------------------------------

chroma_db_path = get_chroma_db_path()
client = chromadb.PersistentClient(path=chroma_db_path)
collection = client.get_collection("reg_chunks")

# ------------------------------
# GRAPH-AWARE RETRIEVAL
# ------------------------------

def get_neighbors(chunk_id):
    obj = kg_by_chunk.get(chunk_id)
    if not obj:
        return []

    ents = {e["local_id"]: e for e in obj["entities"]}
    neighbors = []

    for rel in obj["relations"]:
        s = ents.get(rel["source_local_id"])
        t = ents.get(rel["target_local_id"])
        if not s or not t:
            continue
        neighbors.append({
            "source": s,
            "target": t,
            "type": rel["type"]
        })

    return neighbors

def graph_aware_retrieve(query, top_k=5):
    q_emb = embed(query)

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k
    )

    hits = []
    for i in range(len(res["ids"][0])):
        text = res["documents"][0][i]
        meta = res["metadatas"][0][i]
        chunk_id = meta["chunk_id"]

        neighbors = get_neighbors(chunk_id)
        community_id = chunk_to_community.get(chunk_id)

        hits.append({
            "text": text,
            "meta": meta,
            "neighbors": neighbors,
            "community_id": community_id
        })

    return hits

# ------------------------------
# COMMUNITY SUMMARIES
# ------------------------------

def get_community_summary(cid):
    c = community_by_id.get(cid)
    if not c:
        return None
    return c["metadata"]

# ------------------------------
# STRICT OBLIGATION-FOCUSED ANSWER
# ------------------------------

STRICT_PROMPT = """
You are a regulatory compliance analyst.

Your task:
- Extract ONLY obligations, duties, prohibitions, and requirements.
- Focus specifically on the actor mentioned in the user question.
- Use ONLY the provided text, community summaries, and graph facts.
- If the text does not support a claim, say "Not supported by retrieved text."

Format:
1. Direct obligations
2. Conditions / exceptions
3. Related actors (from graph facts)
4. Citations (chunk IDs)
"""

def ask(query, top_k=5):
    hits = graph_aware_retrieve(query, top_k=top_k)

    # Print internals for debugging
    print("\n=== Retrieved Chunks ===")
    for h in hits:
        print(f"- {h['meta']['chunk_id']}")

    print("\n=== Graph Facts ===")
    for h in hits:
        for n in h["neighbors"]:
            print(f"{n['source']['name']} -[{n['type']}]-> {n['target']['name']}")

    # Build context
    chunk_texts = "\n\n".join(
        f"[{h['meta']['chunk_id']}]\n{h['text']}" for h in hits
    )

    community_ids = {h["community_id"] for h in hits if h["community_id"] is not None}
    community_text = "\n\n".join(
        f"[Community {cid}] {get_community_summary(cid)}" for cid in community_ids
    )

    graph_facts = "\n".join(
        f"{n['source']['name']} -[{n['type']}]-> {n['target']['name']}"
        for h in hits for n in h["neighbors"]
    )

    prompt = f"""
{STRICT_PROMPT}

User question:
{query}

Retrieved text:
{chunk_texts}

Community summaries:
{community_text}

Graph facts:
{graph_facts}

Provide a structured, obligation-focused answer.
"""

    answer = llm(prompt)
    return answer

# ------------------------------
# Example
# ------------------------------

if __name__ == "__main__":
    QUESTION = "What obligations does the EU AI Act impose on Providers?" 
    print(f"Question: {QUESTION}")
    print("\nAnswer:")
    print(ask("What obligations does the EU AI Act impose on Providers?"))