import json
from pathlib import Path
from collections import Counter

# repair_jsonl("./docs/reg_kg_triples.jsonl", "./graph-rag-semantic/reg_kg_triples_repaired.jsonl")
INPUT_PATH = Path("./docs/reg_kg_triples.jsonl")
OUTPUT_PATH = Path("./docs/reg_kg_triples_repaired.jsonl")
LOG_PATH = Path("reg_kg_triples_repair_log.json")

ALLOWED_ENTITY_TYPES = {
    "AI_System",
    "High_Risk_System",
    "Provider",
    "User",
    "Regulation",
    "Obligation",
    "Process",
    "Authority",
}

ALLOWED_RELATION_TYPES = {
    "REQUIRES",
    "MUST_COMPLY_WITH",
    "IS_DEFINED_IN",
    "IS_PART_OF",
    "APPLIES_TO",
    "SUPERVISED_BY",
}

def infer_regulation_from_text(text: str):
    """Very simple heuristic; extend as needed."""
    t = text.lower()
    if "eu ai act" in t:
        return "EU AI Act"
    if "swiss fadp" in t:
        return "Swiss FADP"
    if "coe ai convention" in t:
        return "CoE AI Convention"
    if "ai act" in t:
        return "EU AI Act"
    return None

def repair_record(obj, stats):
    entities = obj.get("entities", [])
    relations = obj.get("relations", [])
    text = obj.get("text", "")

    # 1) Drop malformed entities
    cleaned_entities = []
    for e in entities:
        if not all(k in e for k in ("local_id", "name", "type")):
            stats["dropped_entities_malformed"] += 1
            continue
        if e["type"] not in ALLOWED_ENTITY_TYPES:
            stats["dropped_entities_invalid_type"] += 1
            continue
        cleaned_entities.append(e)

    # Rebuild id set
    id_set = {e["local_id"] for e in cleaned_entities}

    # 2) Ensure at least one Regulation entity if inferable
    has_regulation = any(e["type"] == "Regulation" for e in cleaned_entities)
    if not has_regulation:
        inferred_name = infer_regulation_from_text(text)
        if inferred_name is not None:
            new_id = "reg1"
            # Avoid collision
            suffix = 1
            while new_id in id_set:
                suffix += 1
                new_id = f"reg{suffix}"
            cleaned_entities.append({
                "local_id": new_id,
                "name": inferred_name,
                "type": "Regulation",
            })
            id_set.add(new_id)
            stats["injected_regulation_entities"] += 1

    # 3) Drop malformed / invalid relations
    cleaned_relations = []
    for r in relations:
        if not all(k in r for k in ("source_local_id", "target_local_id", "type")):
            stats["dropped_relations_malformed"] += 1
            continue
        if r["type"] not in ALLOWED_RELATION_TYPES:
            stats["dropped_relations_invalid_type"] += 1
            continue
        if r["source_local_id"] not in id_set:
            stats["dropped_relations_missing_source"] += 1
            continue
        if r["target_local_id"] not in id_set:
            stats["dropped_relations_missing_target"] += 1
            continue
        cleaned_relations.append(r)

    obj["entities"] = cleaned_entities
    obj["relations"] = cleaned_relations

    if not cleaned_entities:
        stats["chunks_no_entities_after_repair"] += 1
    if not cleaned_relations:
        stats["chunks_no_relations_after_repair"] += 1

    return obj

def main():
    stats = Counter()
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input JSONL not found: {INPUT_PATH}")

    with INPUT_PATH.open("r", encoding="utf-8") as fin, \
         OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["lines_invalid_json"] += 1
                continue

            repaired = repair_record(obj, stats)
            fout.write(json.dumps(repaired, ensure_ascii=False) + "\n")

    with LOG_PATH.open("w", encoding="utf-8") as flog:
        json.dump(stats, flog, indent=2)

    print("Repair finished.")
    print("Stats:")
    for k, v in stats.items():
        print(f" - {k}: {v}")

if __name__ == "__main__":
    main()
# 