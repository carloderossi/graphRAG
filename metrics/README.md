# GraphRAG Metrics — Chroma vs Neo4j Comparison

**Framing question:** *When is a graph DB worth the operational overhead?*

---

## Files in this folder

| File | Purpose |
|---|---|
| `ground_truth_obligations.json` | 28 curated obligations from EU AI Act (14), Swiss FADP (8), Swiss AI Guidelines (6) |
| `test_queries.json` | 22 benchmark questions in 4 tiers, each mapped to a ground-truth obligation |
| `judge_harness.py` | Full comparison runner — retrieves, asks, judges, logs |
| `comparison_log.jsonl` | Append-only per-question result log (created at runtime) |
| `scores_summary.json` | Aggregated win/loss/draw table by tier (created at runtime) |

---

## Obligation tiers

| Tier | Label | Count | Expected winner |
|---|---|---|---|
| A | Flat retrieval | 16 | Draw |
| B | Thematic / community | 8 | Chroma (if community search wired up) |
| C | Multi-hop relational | 3 | Neo4j |
| D | Cross-document bridge | 1 | Both struggle — most informative |

---

## Running the comparison

```bash
# Full run — all 28 obligations
cd C:\Carlo\projects\graphRAG\graph-rag-compare
python metrics/judge_harness.py

# Run only Tier C questions (the Neo4j showcase)
python metrics/judge_harness.py EU_AIA_007 EU_AIA_009 EU_AIA_014

# Single question for fast iteration
python metrics/judge_harness.py EU_AIA_007
```

**Prerequisites:**
- Ollama running with `llama3.1:8b` and `mxbai-embed-large` loaded
- ChromaDB persisted at `graph-rag-semantic/chroma` (run `pipeline.py` then `create_chroma_db.cmd`)
- Neo4j running at `bolt://localhost:7687` with data imported (run `queries.cypher`)
- KG triples at `docs/reg_kg_triples_v2.jsonl`

---

## Judge dimensions (0–5 each, max 20 per question)

| Dimension | What it measures |
|---|---|
| `faithfulness` | Every claim traceable to retrieved text |
| `obligation_coverage` | Key obligation from ground truth is surfaced |
| `multi_hop_depth` | Answer chains across entities / articles / documents |
| `no_hallucination` | No invented articles, actors, or obligations |

---

## What each tier reveals

### Tier A — control group
Both systems should score similarly (±1 point). High Jaccard (chunk overlap ≥ 0.6) expected since both use the same embeddings. If scores diverge here, investigate prompt or retrieval differences, not graph structure.

### Tier B — community search advantage
Chroma wins **only** if the two-tier retrieval is wired up. Required fix in `graphrag_query.py`:

```python
def community_route(q_emb, communities):
    best_cid, best_score = None, -1
    for c in communities:
        c_emb = np.array(c["community_embedding"])
        score = np.dot(q_emb, c_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(c_emb))
        if score > best_score:
            best_score, best_cid = score, c["community_id"]
    return best_cid

# Then filter Chroma query to only that community's chunks:
member_ids = community_by_id[best_cid]["member_ids"]
res = collection.query(query_embeddings=[q_emb], n_results=TOP_K,
                       where={"chunk_id": {"$in": member_ids}})
```

### Tier C — the Neo4j showcase
**This is where the overhead justification lives.** Neo4j's multi-hop Cypher traverses:

```
Provider → OBLIGATION → QualityManagementSystem → INCLUDES → PostMarketMonitoring
```

The Chroma KG sidecar only does single-hop per-chunk lookup — it cannot cross chunk boundaries. Neo4j can, because entities are globally scoped in the graph. The `neo4j_avg_graph_facts` metric will be significantly higher than `chroma_avg_graph_facts` for Tier C questions. That structural gap is the visual centrepiece of the showcase.

### Tier D — the gap both systems share
Neither system merges entities across documents. `EU_AI_Act_47:e3` ("Provider") and `Swiss_FADP_12:e1` ("Controller") are different nodes even though they conceptually overlap.

**The showcase insight:** Neo4j is architecturally capable of solving this with a one-line import change:

```cypher
-- Current (scoped per chunk):
MERGE (ent:Entity {id: chunk_id + ':' + e.local_id})

-- Fixed (globally merged by canonical name+type):
MERGE (ent:Entity {name: e.name, type: e.type})
```

This would make `Provider` a single node connected to obligations across all 681 chunks — enabling true cross-document multi-hop. ChromaDB's flat JSON sidecar cannot replicate this without rebuilding the architecture. This is the strongest argument for Neo4j as a first-class architectural choice rather than an operational cost.

---

## Key output columns to watch

| Column | Tier A expectation | Tier C expectation |
|---|---|---|
| `chroma_avg_score` | ≈ `neo4j_avg_score` | Lower |
| `neo4j_avg_score` | ≈ `chroma_avg_score` | Higher |
| `avg_chunk_overlap_jaccard` | ≥ 0.6 | May be lower (different traversal paths) |
| `chroma_avg_graph_facts` | Small (5–15) | Small (5–15, still single-hop) |
| `neo4j_avg_graph_facts` | Small (5–15) | Large (20–40, multi-hop paths) |

The `neo4j_avg_graph_facts >> chroma_avg_graph_facts` gap in Tier C is the headline metric.
It shows that the graph DB surfaces structurally richer context, independently of whether the LLM synthesises it well.
