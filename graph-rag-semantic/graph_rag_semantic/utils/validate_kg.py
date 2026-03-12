import json, os
from pathlib import Path
from collections import defaultdict

def validate_jsonl(path):
    errors = []
    seen_entity_ids = set()
    if not os.path.exists(path):
        raise Exception(f"Could not find JSONL file '{path}'")
    
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"[Line {line_no}] Invalid JSON: {e}")
                continue

            # Required fields
            for field in ["chunk_id", "source", "text", "entities", "relations"]:
                if field not in obj:
                    errors.append(f"[Line {line_no}] Missing field: {field}")

            # Validate entities
            local_ids = set()
            for ent in obj.get("entities", []):
                if "local_id" not in ent or "name" not in ent or "type" not in ent:
                    errors.append(f"[Line {line_no}] Malformed entity: {ent}")
                else:
                    if ent["local_id"] in local_ids:
                        errors.append(f"[Line {line_no}] Duplicate local_id in chunk: {ent['local_id']}")
                    local_ids.add(ent["local_id"])

            # Validate relations
            for rel in obj.get("relations", []):
                for field in ["source_local_id", "target_local_id", "type"]:
                    if field not in rel:
                        errors.append(f"[Line {line_no}] Malformed relation: {rel}")
                if rel.get("source_local_id") not in local_ids:
                    errors.append(f"[Line {line_no}] Relation source missing: {rel}")
                if rel.get("target_local_id") not in local_ids:
                    errors.append(f"[Line {line_no}] Relation target missing: {rel}")

    if errors:
        print("\n❌ Structural issues found:")
        for e in errors:
            print(" -", e)
    else:
        print("✅ No structural issues detected.")

#validate_jsonl("./docs/reg_kg_triples.jsonl")
validate_jsonl("./graph-rag-semantic/reg_kg_triples_repaired.jsonl")