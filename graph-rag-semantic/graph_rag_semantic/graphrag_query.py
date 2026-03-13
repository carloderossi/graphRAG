import json
from collections import defaultdict
import chromadb
import ollama

from utils.config import (
    CHROMA_PATH,
    SEMANTIC_INDEX_PATH,
    KG_PATH,
    EMBED_MODEL,
    LLM_MODEL,
)

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

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection("reg_chunks")

# ------------------------------
# GRAPH-AWARE RETRIEVAL
# ------------------------------

OBLIGATION_REL_TYPES = {
    "MUST_COMPLY_WITH",
    "REQUIRES",
    "PROHIBITED",
    "IS_PART_OF",
    "APPLIES_TO",
}

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

def filter_graph_facts(neighbors, actor_terms):
    facts = []
    for n in neighbors:
        if n["type"] not in OBLIGATION_REL_TYPES:
            continue
        s_name = n["source"]["name"]
        t_name = n["target"]["name"]
        if any(a.lower() in s_name.lower() or a.lower() in t_name.lower() for a in actor_terms):
            facts.append(f"{s_name} -[{n['type']}]-> {t_name}")
    return facts

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
    return c.get("metadata", {})

# ------------------------------
# STRICT OBLIGATION-FOCUSED PROMPT
# ------------------------------

STRICT_PROMPT = """
You are a regulatory compliance analyst.

Use ONLY:
- Retrieved chunks
- Community summaries
- Graph facts

Task:
- Extract ONLY obligations, duties, prohibitions, and requirements.
- Focus on the actor(s) mentioned in the user question.
- Do NOT invent obligations.
- If something is not supported by the text, say: "Not supported by retrieved text."

Format:
1. Direct obligations (bullet list)
2. Conditions / exceptions
3. Related actors (from graph facts)
4. Citations (chunk IDs)
"""

def extract_actor_terms(question: str):
    # very simple heuristic; you can extend this
    candidates = ["provider", "user", "deployer", "high-risk", "ai system", "swiss actors"]
    return [c for c in candidates if c.lower() in question.lower()]

def ask(query, top_k=5, debug=True):
    actor_terms = extract_actor_terms(query)
    hits = graph_aware_retrieve(query, top_k=top_k)

    if debug:
        print("\n=== Retrieved Chunks ===")
        for h in hits:
            print(f"- {h['meta']['chunk_id']}")

    # community summaries
    community_ids = {h["community_id"] for h in hits if h["community_id"] is not None}
    community_summaries = {
        cid: get_community_summary(cid) for cid in community_ids
    }

    # graph facts (filtered)
    all_graph_facts = []
    for h in hits:
        facts = filter_graph_facts(h["neighbors"], actor_terms or ["provider", "user", "deployer"])
        all_graph_facts.extend(facts)

    if debug:
        print("\n=== Graph Facts (filtered) ===")
        for f in all_graph_facts:
            print(f)

    # build context
    chunk_texts = "\n\n".join(
        f"[{h['meta']['chunk_id']}]\n{h['text']}" for h in hits
    )

    community_text = "\n\n".join(
        f"[Community {cid}] title={meta.get('title')}\nsummary={meta.get('summary')}\nfindings={meta.get('findings')}"
        for cid, meta in community_summaries.items()
    )

    graph_text = "\n".join(all_graph_facts)

    prompt = f"""
{STRICT_PROMPT}

User question:
{query}

Retrieved text:
{chunk_texts}

Community summaries:
{community_text}

Graph facts:
{graph_text}

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