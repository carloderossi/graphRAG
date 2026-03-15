import os
import traceback
import json
from pathlib import Path
from typing import Dict, Any
import ollama

# --- CONFIGURATION ---
MODELS = {
    "embed": "mxbai-embed-large:latest",  # unchanged
    "slm": "llama3.1:8b"
}

AI_REG_INDEX_PATH = Path("./ai_reg_semantic_index.json")
OUTPUT_JSONL_PATH = Path("reg_kg_triples.jsonl")

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

# --- PROMPTS (compressed, 2048-safe) ---

SYSTEM_PROMPT = (
    "You are an expert in European and Swiss AI regulation. "
    "Extract entities and relations from legal text using the provided schema and ontology."
)

EXTRACTION_PROMPT_TEMPLATE = """
You extract a knowledge graph from legal text (EU AI Act, Swiss FADP, Swiss AI Guidelines, CoE AI Convention).

Schema:
{{
  "entities": [{{"local_id": "e1", "name": "...", "type": "..."}}],
  "relations": [{{"source_local_id": "e1", "target_local_id": "e2", "type": "..."}}]
}}

Entity types (pick closest):
Actor, Authority, AI_System, Regulation, Obligation, Process, Concept, Document, Article, Chapter

Relation types:
OBLIGATION, COMPLIANCE, GOVERNANCE, SUPERVISION, DEFINITION, STRUCTURAL, REFERENCE, RELATED_TO

Rules:
- Always include the regulation entity (e.g. "EU AI Act", "Swiss FADP", "CoE AI Convention", "Swiss AI Guidelines") when clear.
- Extract obligations even if implicit (“shall”, “must”, “is required to”).
- Extract actor–action–object structure (who does what to whom/what).
- Always produce at least one relation if any interaction exists.
- Use RELATED_TO if no specific type fits.
- Use short canonical names.

Return only:
{{
  "entities": [...],
  "relations": [...]
}}

Text:
\"\"\"{chunk_text}\"\"\"
"""


def call_llm_for_kg(chunk_text: str) -> Dict[str, Any]:
    """
    Call the SLM and return a dict with keys: entities, relations.
    Uses compressed prompts and schema-constrained JSON.
    """
    user_prompt = EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunk_text)

    try:
        response = ollama.generate(
            model=MODELS["slm"],
            prompt=f"{SYSTEM_PROMPT}\n\n{user_prompt}",
            format=KG_SCHEMA,  # schema-constrained JSON
            options={"temperature": 0}
        )

        parsed = json.loads(response["response"])
        entities = parsed.get("entities", []) or []
        relations = parsed.get("relations", []) or []

        # Safety: ensure lists
        if not isinstance(entities, list):
            entities = []
        if not isinstance(relations, list):
            relations = []

        return {
            "entities": entities,
            "relations": relations,
        }

    except Exception as e:
        print("Error during KG extraction:")
        traceback.print_exc()
        # In case of failure, return empty structures to keep pipeline running
        return {
            "entities": [],
            "relations": [],
        }


def load_chunks(path: Path) -> Dict[str, Dict[str, Any]]:
    print(f"Loading chunks from {path}...")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["chunks"]


def main():
    print("\n=== 1. LOAD CHUNKS ===")
    chunks = load_chunks(AI_REG_INDEX_PATH)
    print(f"Successfully loaded {len(chunks)} chunks.")

    print("Calling SLM for generating Knowledge Graph...\n", end=" ", flush=True)

    i = 0
    with OUTPUT_JSONL_PATH.open("w", encoding="utf-8") as out_f:
        for chunk_id, c in chunks.items():
            text = c.get("text", "")
            source = c.get("source", "")

            if i % 10 == 0:
                print("•", end="", flush=True)
            else:
                print(".", end="", flush=True)

            kg = call_llm_for_kg(text)

            record = {
                "chunk_id": chunk_id,
                "source": source,
                "text": text,
                "entities": kg.get("entities", []),
                "relations": kg.get("relations", []),
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            i += 1

    print(f"\nWrote KG triples to {OUTPUT_JSONL_PATH}.")


if __name__ == "__main__":
    main()