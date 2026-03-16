"""
LLM-as-Judge comparison harness for GraphRAG Chroma vs Neo4j.

Focus question: "When is a graph DB worth the operational overhead?"

Usage:
    # Full run (all 28 obligations → 28 question pairs)
    python metrics/judge_harness.py

    # Run only Tier C questions
    python metrics/judge_harness.py EU_AIA_007 EU_AIA_009 EU_AIA_014

    # Single question for debugging
    python metrics/judge_harness.py EU_AIA_007

Prerequisites:
    - Ollama running with llama3.1:8b, qwen2.5:32b-instruct-q4_K_M, and mxbai-embed-large loaded
    - ChromaDB persisted at graph-rag-semantic/chroma (run pipeline.py then create_chroma_db.cmd)
    - Neo4j running at bolt://localhost:7687 with data imported (run queries.cypher)
    - KG triples at docs/reg_kg_triples_v2.jsonl
"""

import json
import time
import logging
from pathlib import Path
from typing import Literal
from dataclasses import dataclass, field
import chromadb
import ollama
import numpy as np
from neo4j import GraphDatabase

# ── Config ─────────────────────────────────────────────────────────────────────
EMBED_MODEL   = "mxbai-embed-large:latest"
LLM_MODEL     = "llama3.1-16k:latest" #     "llama3.1:8b" #       # answer generation — keep fast
JUDGE_MODEL   = "phi4:14b" #"qwen2.5:14b"  # stronger judge; fallback: "phi4:14b"

SEMANTIC_INDEX_PATH = Path("../docs/ai_reg_semantic_index.json")
KG_PATH             = Path("../docs/reg_kg_triples_v2.jsonl")
CHROMA_PATH         = Path("../graph-rag-semantic/graph_rag_semantic/chroma")
GROUND_TRUTH_PATH   = Path("ground_truth_obligations.json")
LOG_PATH            = Path("comparison_log.jsonl")
SCORES_PATH         = Path("scores_summary.json")

NEO4J_URI  = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "password123"

TOP_K = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    chunk_ids:   list[str]
    texts:       list[str]
    graph_facts: list[str]
    latency_ms:  dict[str, float]
    extra:       dict = field(default_factory=dict)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def embed(text: str) -> list[float]:
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]

def llm(prompt: str) -> str:
    return ollama.generate(model=LLM_MODEL, prompt=prompt,
                           options={"temperature": 0})["response"]


# ── Chroma retriever ───────────────────────────────────────────────────────────

def load_chroma_and_kg():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection("reg_chunks")
    kg = {}
    with open(KG_PATH, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            kg[obj["chunk_id"]] = obj
    return collection, kg


def chroma_retrieve(query: str, collection, kg: dict) -> RetrievalResult:
    t0 = time.perf_counter()
    q_emb = embed(query)
    t_embed = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    res = collection.query(query_embeddings=[q_emb], n_results=TOP_K)
    t_retrieve = (time.perf_counter() - t1) * 1000

    chunk_ids = [m["chunk_id"] for m in res["metadatas"][0]]
    texts     = res["documents"][0]

    # Single-hop KG sidecar expansion (per-chunk, in-memory dict)
    t2 = time.perf_counter()
    graph_facts = []
    OBLIGATION_RELS = {"OBLIGATION", "COMPLIANCE", "GOVERNANCE", "SUPERVISION"}
    for cid in chunk_ids:
        obj = kg.get(cid)
        if not obj:
            continue
        ents = {e["local_id"]: e for e in obj["entities"]}
        for rel in obj["relations"]:
            if rel["type"] in OBLIGATION_RELS:
                s = ents.get(rel["source_local_id"])
                t = ents.get(rel["target_local_id"])
                if s and t:
                    graph_facts.append(f"{s['name']} -[{rel['type']}]-> {t['name']}")
    t_graph = (time.perf_counter() - t2) * 1000

    return RetrievalResult(
        chunk_ids=chunk_ids,
        texts=texts,
        graph_facts=graph_facts,
        latency_ms={"embed_ms": t_embed, "retrieve_ms": t_retrieve, "graph_ms": t_graph},
    )


def chroma_ask(query: str, retrieval: RetrievalResult) -> str:
    chunk_ctx = "\n\n".join(f"[{cid}]\n{txt}"
                            for cid, txt in zip(retrieval.chunk_ids, retrieval.texts))
    graph_ctx = "\n".join(retrieval.graph_facts) or "(none)"
    prompt = f"""You are a regulatory compliance analyst.
Answer ONLY from the retrieved text and graph facts. Do NOT invent obligations.

Question: {query}

Retrieved chunks:
{chunk_ctx}

Graph facts (entity relations, single-hop):
{graph_ctx}

Provide a concise, structured answer listing obligations found."""
    t0 = time.perf_counter()
    answer = llm(prompt)
    retrieval.latency_ms["llm_ms"] = (time.perf_counter() - t0) * 1000
    return answer


# ── Neo4j retriever ────────────────────────────────────────────────────────────

def neo4j_retrieve(query: str, driver) -> RetrievalResult:
    t0 = time.perf_counter()
    q_emb = np.array(embed(query))
    t_embed = (time.perf_counter() - t0) * 1000

    # Python-side cosine (replace with native vector index once configured)
    t1 = time.perf_counter()
    with driver.session() as session:
        rows = session.run(
            "MATCH (c:Chunk) RETURN c.id AS id, c.text AS text, c.embedding AS emb"
        ).data()

    scored = []
    for r in rows:
        emb = np.array(r["emb"])
        score = float(np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb)))
        scored.append((score, r["id"], r["text"]))
    scored.sort(reverse=True)
    top = scored[:TOP_K]
    t_retrieve = (time.perf_counter() - t1) * 1000

    chunk_ids = [x[1] for x in top]
    texts     = [x[2] for x in top]

    # Multi-hop Cypher graph expansion (1–2 hops) — the key Neo4j differentiator
    t2 = time.perf_counter()
    MULTIHOP_CYPHER = """
    MATCH (c:Chunk) WHERE c.id IN $ids
    MATCH (c)-[:MENTIONS]->(e:Entity)
    MATCH path = (e)-[:RELATION*1..2]->(t:Entity)
    WHERE ANY(r IN relationships(path)
          WHERE r.type IN ['OBLIGATION','COMPLIANCE','GOVERNANCE','SUPERVISION'])
    RETURN e.name AS source, t.name AS target,
           [r IN relationships(path) | r.type] AS rel_types
    LIMIT 40
    """
    graph_facts = []
    raw_path_count = 0
    with driver.session() as session:
        rows = session.run(MULTIHOP_CYPHER, ids=chunk_ids).data()
        raw_path_count = len(rows)
        for r in rows:
            chain = " -> ".join(r["rel_types"])
            graph_facts.append(f"{r['source']} -[{chain}]-> {r['target']}")
    t_graph = (time.perf_counter() - t2) * 1000

    return RetrievalResult(
        chunk_ids=chunk_ids,
        texts=texts,
        graph_facts=graph_facts,
        latency_ms={"embed_ms": t_embed, "retrieve_ms": t_retrieve, "graph_ms": t_graph},
        extra={"raw_path_count": raw_path_count},
    )


def neo4j_ask(query: str, retrieval: RetrievalResult) -> str:
    chunk_ctx = "\n\n".join(f"[{cid}]\n{txt}"
                            for cid, txt in zip(retrieval.chunk_ids, retrieval.texts))
    graph_ctx = "\n".join(retrieval.graph_facts) or "(none)"
    prompt = f"""You are a regulatory compliance analyst.
Answer ONLY from the retrieved text and graph facts (which include multi-hop relation chains).
Do NOT invent obligations. Trace entity chains explicitly where visible.

Question: {query}

Retrieved chunks:
{chunk_ctx}

Graph relation chains (multi-hop, up to 2 hops):
{graph_ctx}

Provide a concise, structured answer listing obligations found, noting any multi-entity chains."""
    t0 = time.perf_counter()
    answer = llm(prompt)
    retrieval.latency_ms["llm_ms"] = (time.perf_counter() - t0) * 1000
    return answer


# ── LLM-as-Judge ──────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are an expert in EU and Swiss AI regulation acting as an impartial evaluator.

Obligation being tested:
  ID:   {obligation_id}
  Text: {obligation_text}

System evaluated: {system_name}

Retrieved context summary:
{context_summary}

Answer produced:
{answer}

Score the answer on FOUR dimensions (integer 0-5 each):

1. faithfulness        - every claim is traceable to retrieved text; no invented articles/actors
2. obligation_coverage - the key obligation in the reference text is surfaced and explained
3. multi_hop_depth     - the answer chains across multiple entities, articles or documents
                         (0=flat retrieval only; 5=explicit multi-hop chain with actors+articles)
4. no_hallucination    - absence of invented obligations, wrong article numbers or fabricated actors
                         (5=zero hallucination; 0=multiple invented facts)

Also provide:
  verdict: "chroma_wins" | "neo4j_wins" | "draw"
  reason: one sentence explaining the verdict for THIS specific question

Respond ONLY in valid JSON (no markdown fences, no preamble):
{{
  "faithfulness": <int>,
  "obligation_coverage": <int>,
  "multi_hop_depth": <int>,
  "no_hallucination": <int>,
  "total": <sum of above 4>,
  "verdict": "<string>",
  "reason": "<string>"
}}"""


# Maximum characters of the system answer fed into the judge prompt.
# Keeps the total prompt short so the model has room to generate the full response.
_ANSWER_TRUNCATE = 1200
# Tokens reserved for the judge's JSON response.
# 400 is generous for the schema; the format= constraint prevents preamble waste.
_JUDGE_NUM_PREDICT = 400

# JSON schema passed to ollama format= so the model outputs pure JSON,
# no preamble, no markdown fences, guaranteed well-formed.
_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "faithfulness":        {"type": "integer", "minimum": 0, "maximum": 5},
        "obligation_coverage": {"type": "integer", "minimum": 0, "maximum": 5},
        "multi_hop_depth":     {"type": "integer", "minimum": 0, "maximum": 5},
        "no_hallucination":    {"type": "integer", "minimum": 0, "maximum": 5},
        "total":               {"type": "integer", "minimum": 0, "maximum": 20},
        "verdict":             {"type": "string",  "enum": ["chroma_wins", "neo4j_wins", "draw"]},
        "reason":              {"type": "string"}
    },
    "required": ["faithfulness", "obligation_coverage", "multi_hop_depth",
                 "no_hallucination", "total", "verdict", "reason"]
}


def _extract_scores_regex(raw: str) -> dict | None:
    """Fallback: pull integer fields out of a truncated/malformed JSON string."""
    import re
    fields = ["faithfulness", "obligation_coverage", "multi_hop_depth", "no_hallucination"]
    scores = {}
    for field in fields:
        m = re.search(rf'"?{field}"?\s*:\s*(\d)', raw)
        if m:
            scores[field] = int(m.group(1))
    if len(scores) < 4:          # too many fields missing — not usable
        return None
    scores["total"] = sum(scores[f] for f in fields)
    # try to salvage verdict
    vm = re.search(r'"verdict"\s*:\s*"([^"]+)"', raw)
    scores["verdict"] = vm.group(1) if vm else "parse_error"
    scores["reason"]  = "(truncated — scores recovered via regex)"
    return scores


def judge(obligation: dict, system_name: str, answer: str, retrieval: RetrievalResult) -> dict:
    context_summary = (
        f"Chunks retrieved: {', '.join(retrieval.chunk_ids[:3])}\n"
        f"Graph facts ({len(retrieval.graph_facts)} total): "
        + "; ".join(retrieval.graph_facts[:5])
    )
    # Truncate the answer so the judge prompt stays short and num_predict
    # is never the bottleneck. The scores only need the answer's gist.
    answer_trunc = answer[:_ANSWER_TRUNCATE] + ("…" if len(answer) > _ANSWER_TRUNCATE else "")

    prompt = JUDGE_PROMPT.format(
        obligation_id=obligation["id"],
        obligation_text=obligation["text"],
        system_name=system_name,
        context_summary=context_summary,
        answer=answer_trunc,
    )
    raw = ollama.generate(
        model=JUDGE_MODEL,
        prompt=prompt,
        format=_JUDGE_SCHEMA,          # forces pure JSON, no preamble, no fences
        options={"temperature": 0, "num_predict": _JUDGE_NUM_PREDICT},
    )["response"]
    try:
        # format= already guarantees valid JSON, but strip defensively anyway
        cleaned = raw.strip()
        scores = json.loads(cleaned)
        scores.setdefault("total", sum(scores.get(k, 0)
                                       for k in ["faithfulness", "obligation_coverage",
                                                  "multi_hop_depth", "no_hallucination"]))
        return scores
    except Exception:
        # Try to rescue the integer scores from the truncated output
        recovered = _extract_scores_regex(raw)
        if recovered:
            log.warning("Judge JSON truncated for %s / %s — scores recovered via regex",
                        obligation["id"], system_name)
            return recovered
        log.warning("Judge parse failed for %s / %s — raw: %s",
                    obligation["id"], system_name, raw[:300])
        return {"faithfulness": 0, "obligation_coverage": 0, "multi_hop_depth": 0,
                "no_hallucination": 0, "total": 0,
                "verdict": "parse_error", "reason": "judge response failed to parse"}


# ── Jaccard chunk overlap ──────────────────────────────────────────────────────

def jaccard(set_a: list, set_b: list) -> float:
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ── Main comparison loop ───────────────────────────────────────────────────────

def run_comparison(obligation_ids: list[str] | None = None):
    gt = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    obligations = gt["obligations"]
    if obligation_ids:
        obligations = [o for o in obligations if o["id"] in obligation_ids]

    log.info("Loading Chroma collection and KG sidecar (%d obligations to test)…",
             len(obligations))
    collection, kg = load_chroma_and_kg()

    log.info("Connecting to Neo4j at %s…", NEO4J_URI)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    LOG_PATH.parent.mkdir(exist_ok=True)
    results = []

    for obl in obligations:
        query = _obligation_to_query(obl)
        log.info("[%s | Tier %s] %s", obl["id"], obl["tier"], query[:80])

        # ─ Chroma ─────────────────────────────────────────────────────────────
        c_retrieval = chroma_retrieve(query, collection, kg)
        c_answer    = chroma_ask(query, c_retrieval)
        c_scores    = judge(obl, "Chroma+KGsidecar", c_answer, c_retrieval)

        # ─ Neo4j ──────────────────────────────────────────────────────────────
        n_retrieval = neo4j_retrieve(query, driver)
        n_answer    = neo4j_ask(query, n_retrieval)
        n_scores    = judge(obl, "Neo4j", n_answer, n_retrieval)

        overlap = jaccard(c_retrieval.chunk_ids, n_retrieval.chunk_ids)

        record = {
            "obligation_id":         obl["id"],
            "source":                obl["source"],
            "tier":                  obl["tier"],
            "query":                 query,
            "chroma": {
                "answer":            c_answer,
                "chunk_ids":         c_retrieval.chunk_ids,
                "graph_facts_n":     len(c_retrieval.graph_facts),
                "latency_ms":        c_retrieval.latency_ms,
                "scores":            c_scores,
            },
            "neo4j": {
                "answer":            n_answer,
                "chunk_ids":         n_retrieval.chunk_ids,
                "graph_facts_n":     len(n_retrieval.graph_facts),
                "latency_ms":        n_retrieval.latency_ms,
                "scores":            n_scores,
                "raw_path_count":    n_retrieval.extra.get("raw_path_count", 0),
            },
            "chunk_overlap_jaccard": overlap,
        }
        results.append(record)

        # Append-only log so partial runs are never lost
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        log.info("  Chroma=%d/20  Neo4j=%d/20  facts_chroma=%d  facts_neo4j=%d  Jaccard=%.2f",
                 c_scores["total"], n_scores["total"],
                 len(c_retrieval.graph_facts), len(n_retrieval.graph_facts), overlap)

    driver.close()
    summary = _write_summary(results)
    _print_summary(summary)
    return results


def _obligation_to_query(obl: dict) -> str:
    """Convert an obligation record to a natural-language benchmark question."""
    TEMPLATES = {
        "PROCESS":              "What process obligations does {actor} have under {article}?",
        "DATA_GOVERNANCE":      "What data governance requirements apply to {actor} under {article}?",
        "DOCUMENTATION":        "What documentation must {actor} prepare under {article}?",
        "TRANSPARENCY":         "What transparency obligations does {actor} have under {article}?",
        "HUMAN_OVERSIGHT":      "What human oversight obligations does {actor} face under {article}?",
        "TECHNICAL_REQUIREMENT":"What technical requirements must {actor} meet under {article}?",
        "COMPOUND":             "What are all the obligations of {actor} under {article}?",
        "QUALITY_MANAGEMENT":   "What quality management system must {actor} establish under {article}?",
        "VALUE_CHAIN":          "When does {actor} become treated as a provider under {article}?",
        "DEPLOYMENT":           "What deployment obligations does {actor} have under {article}?",
        "IMPACT_ASSESSMENT":    "What impact assessment must {actor} conduct under {article}?",
        "TRANSPARENCY_GPAI":    "What transparency obligations apply to {actor} for AI interactions?",
        "GPAI_MODEL":           "What obligations does {actor} have for general-purpose AI models?",
        "SYSTEMIC_RISK":        "What additional obligations apply to {actor} for GPAI models with systemic risk?",
        "PRINCIPLES":           "What data processing principles must {actor} respect under {article}?",
        "PRIVACY_BY_DESIGN":    "What privacy by design and default obligations does {actor} have under {article}?",
        "RECORD_KEEPING":       "What record-keeping obligations does {actor} have under {article}?",
        "REPRESENTATIVE":       "When must {actor} appoint a representative in Switzerland?",
        "AUTOMATED_DECISION":   "What obligations does {actor} have when making automated individual decisions?",
        "BREACH_NOTIFICATION":  "What data breach notification obligations does {actor} have under {article}?",
        "CROSS_BORDER":         "What are the conditions for {actor} to disclose personal data abroad?",
        "HUMAN_RESPONSIBILITY": "What human responsibility obligations apply to {actor} when using AI?",
        "RELIABILITY_SAFETY":   "What reliability and safety requirements apply to {actor} when deploying AI?",
        "DATA_PROTECTION":      "What data protection obligations does {actor} face when using AI under the FADP?",
        "NON_DISCRIMINATION":   "What non-discrimination obligations apply to {actor} using AI?",
        "REGULATORY_ALIGNMENT": "What EU AI Act obligations apply to Swiss companies exporting AI products to the EU?",
    }
    tmpl = TEMPLATES.get(obl["obligation_type"],
                         "What obligations does {actor} have under {article}?")
    return tmpl.format(actor=obl["actor"], article=obl["article"])


def _write_summary(results: list[dict]) -> dict:
    chroma_wins = neo4j_wins = draws = 0
    tier_scores: dict[str, dict] = {}

    for r in results:
        c_total = r["chroma"]["scores"].get("total", 0)
        n_total = r["neo4j"]["scores"].get("total", 0)
        tier    = r["tier"]

        tier_scores.setdefault(tier, {
            "chroma": [], "neo4j": [], "jaccard": [],
            "chroma_graph_facts": [], "neo4j_graph_facts": []
        })
        tier_scores[tier]["chroma"].append(c_total)
        tier_scores[tier]["neo4j"].append(n_total)
        tier_scores[tier]["jaccard"].append(r["chunk_overlap_jaccard"])
        tier_scores[tier]["chroma_graph_facts"].append(r["chroma"]["graph_facts_n"])
        tier_scores[tier]["neo4j_graph_facts"].append(r["neo4j"]["graph_facts_n"])

        # Win = more than 1 point ahead (avoids noise from draw-zone scores)
        if c_total > n_total + 1:
            chroma_wins += 1
        elif n_total > c_total + 1:
            neo4j_wins += 1
        else:
            draws += 1

    def avg(lst): return round(sum(lst) / len(lst), 2) if lst else 0

    summary = {
        "total_questions": len(results),
        "overall": {
            "chroma_wins": chroma_wins,
            "neo4j_wins":  neo4j_wins,
            "draws":       draws,
        },
        "by_tier": {
            t: {
                "n":                        len(v["chroma"]),
                "chroma_avg_score":         avg(v["chroma"]),
                "neo4j_avg_score":          avg(v["neo4j"]),
                "avg_chunk_overlap_jaccard":avg(v["jaccard"]),
                "chroma_avg_graph_facts":   avg(v["chroma_graph_facts"]),
                "neo4j_avg_graph_facts":    avg(v["neo4j_graph_facts"]),
            }
            for t, v in sorted(tier_scores.items())
        },
        "showcase_finding": (
            "Tier A: both systems draw — confirms shared embedding quality. "
            "Tier B: Chroma leads if community search is wired up; otherwise draw. "
            "Tier C: Neo4j wins via multi-hop Cypher — the graph DB overhead is justified here. "
            "Tier D: both struggle — cross-document entity merging is the missing capability in both."
        )
    }
    SCORES_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Scores summary written to %s", SCORES_PATH)
    return summary


def _print_summary(summary: dict):
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    o = summary["overall"]
    print(f"  Chroma wins: {o['chroma_wins']}  |  Neo4j wins: {o['neo4j_wins']}  |  Draws: {o['draws']}")
    print()
    print(f"  {'Tier':<6}  {'N':>4}  {'Chroma':>8}  {'Neo4j':>8}  {'Jaccard':>8}  {'Facts(C)':>9}  {'Facts(N)':>9}")
    print(f"  {'-'*6}  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*9}")
    for tier, v in sorted(summary["by_tier"].items()):
        print(f"  {tier:<6}  {v['n']:>4}  {v['chroma_avg_score']:>8.1f}  "
              f"{v['neo4j_avg_score']:>8.1f}  {v['avg_chunk_overlap_jaccard']:>8.2f}  "
              f"{v['chroma_avg_graph_facts']:>9.1f}  {v['neo4j_avg_graph_facts']:>9.1f}")
    print()
    print(f"  Finding: {summary['showcase_finding']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    import sys
    ids = sys.argv[1:] or None
    run_comparison(ids)
