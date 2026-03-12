import json
from collections import defaultdict
import chromadb
from chromadb.config import Settings
import ollama


# ============================================================
# 1. OLLAMA HELPERS
# ============================================================

EMBED_MODEL = "mxbai-embed-large:latest"
LLM_MODEL = "llama3.1:8b"

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


# ============================================================
# 2. LOAD SEMANTIC INDEX + COMMUNITIES + KG
# ============================================================

def load_semantic_index(path="../docs/ai_reg_semantic_index.json"):
    with open(path, "r", encoding="utf-8") as f:
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


def load_kg(path="../docs/reg_kg_triples_repaired.jsonl"):
    by_chunk = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            by_chunk[obj["chunk_id"]] = obj
    return by_chunk


chunks, communities = load_semantic_index()
kg_by_chunk = load_kg()


# ============================================================
# 3. BUILD CHROMA VECTOR INDEX (CHUNKS)
# ============================================================

client = chromadb.Client(Settings(anonymized_telemetry=False))
chunk_collection = client.create_collection("reg_chunks")

ids, docs, embs, metas = [], [], [], []

for i, item in enumerate(chunks):
    ids.append(str(i))
    docs.append(item["text"])
    embs.append(item["embedding"])
    metas.append({
        "chunk_id": item["chunk_id"],
        "source": item["source"]
    })

chunk_collection.add(
    ids=ids,
    documents=docs,
    embeddings=embs,
    metadatas=metas
)

print(f"Loaded {len(ids)} chunks into Chroma.")


# ============================================================
# 4. COMMUNITY INDEX
# ============================================================

community_by_id = {c["community_id"]: c for c in communities}

# Reverse index: chunk_id → community_id
chunk_to_community = {}
for c in communities:
    for cid in c["member_ids"]:
        chunk_to_community[cid] = c["community_id"]


# ============================================================
# 5. GRAPH-AWARE RETRIEVER
# ============================================================

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


def graph_aware_retrieve(query: str, top_k=5):
    q_emb = embed(query)

    res = chunk_collection.query(
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


# ============================================================
# 6. HIERARCHICAL SUMMARIZATION (COMMUNITY-LEVEL)
# ============================================================

def get_community_summary(community_id):
    if community_id is None:
        return None

    c = community_by_id.get(community_id)
    if not c:
        return None

    meta = c.get("metadata", {})
    title = meta.get("title", "")
    summary = meta.get("summary", "")
    findings = meta.get("findings", [])

    return {
        "title": title,
        "summary": summary,
        "findings": findings
    }


# ============================================================
# 7. FULL PIPELINE: ask(query)
# ============================================================

def ask(query: str, top_k=5):
    hits = graph_aware_retrieve(query, top_k=top_k)

    # Collect community summaries
    community_ids = {h["community_id"] for h in hits if h["community_id"] is not None}
    community_summaries = {
        cid: get_community_summary(cid) for cid in community_ids
    }

    # Graph facts
    graph_facts = []
    for h in hits:
        for n in h["neighbors"]:
            graph_facts.append(
                f"{n['source']['name']} -[{n['type']}]-> {n['target']['name']}"
            )

    # Build final prompt
    chunk_texts = "\n\n".join(h["text"] for h in hits)

    community_text = "\n\n".join(
        f"Community {cid} — {cs['title']}\nSummary: {cs['summary']}\nFindings: {cs['findings']}"
        for cid, cs in community_summaries.items()
    )

    graph_text = "\n".join(graph_facts)

    prompt = f"""
You are a regulatory assistant.

User question:
{query}

Relevant text:
{chunk_texts}

Community summaries:
{community_text}

Graph facts:
{graph_text}

Provide a concise, accurate answer grounded in the regulatory text.
"""

    answer = llm(prompt)
    return answer, hits, community_summaries, graph_facts


# ============================================================
# 8. EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    q = "What obligations does the EU AI Act impose on Providers?"
    answer, hits, communities, facts = ask(q)
    print("\n=== ANSWER ===\n")
    print(answer)