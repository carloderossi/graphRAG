import os
import traceback
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import ollama

# --- CONFIGURATION ---
MODELS = {
    "embed": "mxbai-embed-large:latest",  # unchanged
    "slm": "llama3.1:8b"
}

AI_REG_INDEX_PATH = Path("./ai_reg_semantic_index.json")
OUTPUT_JSONL_PATH = Path("reg_kg_triples_v2.jsonl")

# Keep the JSON schema EXACTLY as-is (for tooling / validation)
KG_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "local_id": {"type": "string"},
                    "name": {"type": "string"},
                    "type": {"type": "string"}
                },
                "required": ["local_id", "name", "type"]
            }
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_local_id": {"type": "string"},
                    "target_local_id": {"type": "string"},
                    "type": {"type": "string"}
                },
                "required": ["source_local_id", "target_local_id", "type"]
            }
        }
    },
    "required": ["entities", "relations"]
}

# --- COMPRESSED ONTOLOGY (for the prompt only) ---

ENTITY_TYPES = [
    "Actor",       # Provider, User, Member State, Deployer, etc.
    "Authority",   # Commission, Board, AI Office, FDPIC, Council of Europe, etc.
    "AI_System",
    "Regulation",
    "Obligation",
    "Process",
    "Concept",
    "Document",
    "Article",
    "Chapter",
]

RELATION_TYPES = [
    "OBLIGATION",   # Actor → Obligation/Process
    "COMPLIANCE",   # Actor/System → Regulation
    "GOVERNANCE",   # Regulation/Authority → Actor/System/Process
    "SUPERVISION",  # Authority → Actor/System
    "DEFINITION",   # Regulation → Concept/Process/Obligation
    "STRUCTURAL",   # Article/Chapter/Process → higher-level structure
    "REFERENCE",    # Article/Regulation → Article/Regulation
    "RELATED_TO",   # fallback when nothing else fits
]

# --- GLOBAL SYSTEM PROMPT (short, 2048-safe) ---

SYSTEM_PROMPT = (
    "You are an expert in European and Swiss AI regulation. "
    "Extract entities and relations from legal text using the provided schema and ontology."
)

# --- 3-PHASE EXTRACTION PROMPTS (compressed) ---
# Phase 1: conservative, schema-first
PHASE1_PROMPT_TEMPLATE = """
You extract a knowledge graph from legal text (EU AI Act, Swiss FADP, Swiss AI Guidelines, CoE AI Convention).

Schema:
{{
  "entities": [{{"local_id": "e1", "name": "...", "type": "..."}}],
  "relations": [{{"source_local_id": "e1", "target_local_id": "e2", "type": "..."}}]
}}

Entity types:
Actor, Authority, AI_System, Regulation, Obligation, Process, Concept, Document, Article, Chapter

Relation types:
OBLIGATION, COMPLIANCE, GOVERNANCE, SUPERVISION, DEFINITION, STRUCTURAL, REFERENCE, RELATED_TO

Phase 1 (core graph):
- Focus on clear actor–obligation–regulation patterns.
- Always include the regulation entity when clear.
- Prefer high-precision relations over recall.
- Use short canonical names.

Return only:
{{
  "entities": [...],
  "relations": [...]
}}

Text:
\"\"\"{chunk_text}\"\"\""""

# Phase 2: aggressive, high-recall inference
PHASE2_PROMPT_TEMPLATE = """
You extend a knowledge graph from the same legal text.

Schema:
{{
  "entities": [{{"local_id": "e1", "name": "...", "type": "..."}}],
  "relations": [{{"source_local_id": "e1", "target_local_id": "e2", "type": "..."}}]
}}

Entity types:
Actor, Authority, AI_System, Regulation, Obligation, Process, Concept, Document, Article, Chapter

Relation types:
OBLIGATION, COMPLIANCE, GOVERNANCE, SUPERVISION, DEFINITION, STRUCTURAL, REFERENCE, RELATED_TO

Phase 2 (high recall):
- Be aggressive: infer obligations, processes, and governance links even if implicit.
- If actors interact, infer OBLIGATION, COMPLIANCE, GOVERNANCE, or SUPERVISION.
- Use RELATED_TO when no specific type fits.
- Do NOT repeat obvious duplicates; extend the graph.

Return only:
{{
  "entities": [...],
  "relations": [...]
}}

Text:
\"\"\"{chunk_text}\"\"\""""

# Phase 3: micro-rescue for still-empty or very sparse chunks
PHASE3_PROMPT_TEMPLATE = """
You perform a last-resort rescue of a knowledge graph from legal text.

Schema:
{{
  "entities": [{{"local_id": "e1", "name": "...", "type": "..."}}],
  "relations": [{{"source_local_id": "e1", "target_local_id": "e2", "type": "..."}}]
}}

Entity types:
Actor, Authority, AI_System, Regulation, Obligation, Process, Concept, Document, Article, Chapter

Relation types:
OBLIGATION, COMPLIANCE, GOVERNANCE, SUPERVISION, DEFINITION, STRUCTURAL, REFERENCE, RELATED_TO

Phase 3 (rescue):
- If any actor, system, authority, or regulation is mentioned, create at least one relation.
- Minimal but non-empty graph: 1–3 entities and 1–3 relations are enough.
- Use RELATED_TO if nothing else fits.

Return only:
{{
  "entities": [...],
  "relations": [...]
}}

Text:
\"\"\"{chunk_text}\"\"\""""

# --- SIMPLE TOKEN ESTIMATOR (approx, to log and keep under 2048) ---
def estimate_tokens(text: str) -> int:
    # Very rough heuristic: ~4 chars per token
    return max(1, len(text) // 4)


def truncate_for_context(chunk_text: str, max_tokens: int = 1400) -> str:
    """
    Truncate chunk_text so that SYSTEM + PROMPT + TEXT stay under ~2048 tokens (ollama limitation: this is for dev env only)
    We keep it simple: cap the text length by tokens.
    """
    # 1400 tokens ~ 5600 chars (~4 chars per token)
    max_chars = max_tokens * 4
    if len(chunk_text) <= max_chars:
        return chunk_text
    return chunk_text[:max_chars]


# --- LLM CALL HELPERS ---
def _call_phase(prompt_template: str, chunk_text: str, phase_name: str) -> Tuple[Dict[str, Any], int]:
    """
    Call one phase, return (parsed_result, approx_total_tokens).
    """
    truncated_text = truncate_for_context(chunk_text)
    user_prompt = prompt_template.format(chunk_text=truncated_text)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    approx_tokens = estimate_tokens(full_prompt)

    response = ollama.generate(
        model=MODELS["slm"],
        prompt=full_prompt,
        format=KG_SCHEMA,
        options={"temperature": 0}
    )

    try:
        parsed = json.loads(response["response"])
    except Exception:
        print(f"\n[WARN] Phase {phase_name}: JSON parse failed, returning empty.")
        parsed = {"entities": [], "relations": []}

    entities = parsed.get("entities", []) or []
    relations = parsed.get("relations", []) or []

    if not isinstance(entities, list):
        entities = []
    if not isinstance(relations, list):
        relations = []

    return {"entities": entities, "relations": relations}, approx_tokens


def call_llm_for_kg_multi_phase(chunk_text: str) -> Dict[str, Any]:
    """
    3-phase KG extraction:
    - Phase 1: conservative, core graph
    - Phase 2: aggressive, high recall (only if relations still empty or very sparse)
    - Phase 3: micro-rescue (only if still empty)
    Logs approximate token usage per phase.
    """
    total_tokens = 0

    # Phase 1
    phase1_result, t1 = _call_phase(PHASE1_PROMPT_TEMPLATE, chunk_text, "P1")
    total_tokens += t1
    entities = phase1_result["entities"]
    relations = phase1_result["relations"]

    # If we already have a decent graph, stop
    if len(relations) >= 2:
        print(f"[P1 ok, tokens≈{total_tokens}]", end="")
        return {"entities": entities, "relations": relations}

    # Phase 2
    phase2_result, t2 = _call_phase(PHASE2_PROMPT_TEMPLATE, chunk_text, "P2")
    total_tokens += t2

    # Merge Phase 2 into Phase 1 (simple concat; you can deduplicate later if needed)
    entities += phase2_result["entities"]
    relations += phase2_result["relations"]

    if len(relations) >= 1:
        print(f"[P1+P2 ok, tokens≈{total_tokens}]", end="")
        return {"entities": entities, "relations": relations}

    # Phase 3 (rescue)
    phase3_result, t3 = _call_phase(PHASE3_PROMPT_TEMPLATE, chunk_text, "P3")
    total_tokens += t3

    entities += phase3_result["entities"]
    relations += phase3_result["relations"]

    print(f"[P1+P2+P3 tokens≈{total_tokens}]", end="")

    return {
        "entities": entities,
        "relations": relations,
    }


# --- CHUNK LOADING & MAIN HARNESS ---
def load_chunks(path: Path) -> Dict[str, Dict[str, Any]]:
    print(f"Loading chunks from {path}...")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["chunks"]


def main():
    print("\n=== LOAD CHUNKS ===")
    chunks = load_chunks(AI_REG_INDEX_PATH)
    print(f"Successfully loaded {len(chunks)} chunks.")

    print("Calling SLM (3-phase) for generating Knowledge Graph...\n", end=" ", flush=True)

    i = 0
    with OUTPUT_JSONL_PATH.open("w", encoding="utf-8") as out_f:
        for chunk_id, c in chunks.items():
            text = c.get("text", "")
            source = c.get("source", "")

            if i % 10 == 0:
                print("•", end="", flush=True)
            else:
                print(".", end="", flush=True)

            kg = call_llm_for_kg_multi_phase(text)

            record = {
                "chunk_id": chunk_id,
                "source": source,
                "text": text,
                "entities": kg.get("entities", []),
                "relations": kg.get("relations", []),
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            i += 1

    print(f"\nWrote KG triples (3-phase) to {OUTPUT_JSONL_PATH}.")


if __name__ == "__main__":
    main()