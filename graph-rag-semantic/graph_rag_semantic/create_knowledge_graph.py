import os
import traceback

import numpy as np
import json

from pathlib import Path
from typing import Dict, Any
import ollama

# --- CONFIGURATION ---
# Using 7B class for reasoning and mxbai-embed-large
MODELS = {
    "embed": "mxbai-embed-large:latest", # nomic-embed-text --> does not support batch embedding 
    "slm": "llama3.1:8b"
}
AI_REG_INDEX_PATH = Path("./ai_reg_semantic_index.json")
OUTPUT_JSONL_PATH = Path("reg_kg_triples.jsonl")

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

def call_llm_for_kg(chunk_text: str) -> Dict[str, Any]:
    """
    Call your LLM and return a dict with keys: entities, relations.
    This is a stub – plug in your actual client (Ollama, Azure, etc.).
    """
    system_prompt = (
        "You are an expert in European and Swiss AI regulation. "
        "You extract a structured knowledge graph from legal text. "
        "Focus on obligations, actors, AI systems, and how they relate to regulations."
    )

    user_prompt = f"""
You are given a text chunk from an AI regulation (EU AI Act, Swiss FADP, Swiss AI Guidelines, CoE AI Convention).

Extract entities and relations according to this schema:

Entity types (type field):
- AI_System
- High_Risk_System
- Provider
- User
- Regulation
- Obligation
- Process
- Authority

Relation types (type field):
- REQUIRES          (Regulation or Obligation → Obligation or Process)
- MUST_COMPLY_WITH  (Provider/User/AI_System/High_Risk_System → Regulation or Obligation)
- IS_DEFINED_IN     (Concept/Entity → Regulation)
- IS_PART_OF        (Obligation/Process → Process)
- APPLIES_TO        (Obligation → AI_System/High_Risk_System/Provider/User)
- SUPERVISED_BY     (Provider/User/AI_System → Authority)

Return a single JSON object with this structure:

{{
  "entities": [
    {{"local_id": "e1", "name": "...", "type": "..."}},
    {{"local_id": "e2", "name": "...", "type": "..."}}
  ],
  "relations": [
    {{"source_local_id": "e1", "target_local_id": "e2", "type": "..."}}
  ]
}}

Rules:
- Use short, canonical names for entities (e.g. "Provider", "User", "High-risk AI system", "EU AI Act", "Swiss FADP").
- If the regulation is clearly EU AI Act, include an entity of type "Regulation" named "EU AI Act".
- If the regulation is clearly Swiss FADP, include an entity of type "Regulation" named "Swiss FADP".
- Only use the relation types listed above.
- If nothing meaningful can be extracted, return empty arrays for entities and relations.

Text chunk:
\"\"\"{chunk_text}\"\"\"
"""

    # --- PSEUDOCODE: replace with your actual LLM call ---
    try:
        # Generate Summary with 7B Model
        response = ollama.generate(
            model=MODELS["slm"], 
            prompt=f"{system_prompt}\n\n{user_prompt}",
            format=KG_SCHEMA,  # schema-constrained JSON
            options={"temperature": 0}
        )
        parsed = json.loads(response["response"])
        entities = parsed.get("entities")
        relations = parsed.get("relations")
        return {
            "entities": entities,
            "relations": relations,
        }
    except Exception as e:
        print(f"Get error for response {parsed}")
        traceback.print_exc()
        # exc_type, exc_value, exc_tb = sys.exc_info()
        # print("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
        raise e        
    # response = llm_client.chat(system=system_prompt, user=user_prompt)
    # parsed = json.loads(response)
    # return parsed
    # ------------------------------------------------------
    #raise NotImplementedError("Implement call_llm_for_kg with your LLM client.")

def load_chunks(path: Path) -> Dict[str, Dict[str, Any]]:
    print(f"Loading chunks from {path}...")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["chunks"]

def main():
    print("\n=== 1. LOAD CHUNKS ===")
    chunks = load_chunks(AI_REG_INDEX_PATH)
    print(f"Successfully loaded {len(chunks)} chunks.")

    i=0
    print("Calling SLM for generating Knowledge Graph...", end=" ", flush=True)
    with OUTPUT_JSONL_PATH.open("w", encoding="utf-8") as out_f:
        for chunk_id, c in chunks.items():
            text = c.get("text", "")
            source = c.get("source", "")
            if i % 10 == 0:
                print("•", end="", flush=True)   # bullet every 10
                # clean up ollama cache
                #ollama.generate(model=MODELS["slm"], prompt="", options={"reset": True})
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
            i=i+1

    print(f"Wrote KG triples to {OUTPUT_JSONL_PATH}")

if __name__ == "__main__":
    main()