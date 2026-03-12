import json
from pathlib import Path

def repair_jsonl(path, out_path):
    repaired = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            obj = json.loads(line)

            entities = obj["entities"]
            relations = obj["relations"]

            # Build lookup maps
            id_map = {e["local_id"]: e for e in entities}
            name_map = {e["name"]: e["local_id"] for e in entities}

            next_id_num = len(entities) + 1

            fixed_relations = []
            for rel in relations:
                src = rel["source_local_id"]
                tgt = rel["target_local_id"]

                # Case 1: numeric IDs → invalid
                if src.isdigit():
                    src = None
                if tgt.isdigit():
                    tgt = None

                # Case 2: name instead of ID
                if src not in id_map and src in name_map:
                    src = name_map[src]
                if tgt not in id_map and tgt in name_map:
                    tgt = name_map[tgt]

                # Case 3: missing entity → auto-create
                if src not in id_map:
                    new_id = f"e{next_id_num}"
                    next_id_num += 1
                    id_map[new_id] = {"local_id": new_id, "name": src, "type": "Unknown"}
                    entities.append(id_map[new_id])
                    src = new_id

                if tgt not in id_map:
                    new_id = f"e{next_id_num}"
                    next_id_num += 1
                    id_map[new_id] = {"local_id": new_id, "name": tgt, "type": "Unknown"}
                    entities.append(id_map[new_id])
                    tgt = new_id

                # Write fixed relation
                fixed_relations.append({
                    "source_local_id": src,
                    "target_local_id": tgt,
                    "type": rel["type"]
                })

            obj["entities"] = entities
            obj["relations"] = fixed_relations
            repaired.append(obj)

    with open(out_path, "w", encoding="utf-8") as f:
        for obj in repaired:
            f.write(json.dumps(obj) + "\n")

    print("✅ Repair completed:", out_path)

repair_jsonl("./docs/reg_kg_triples.jsonl", "./graph-rag-semantic/reg_kg_triples_repaired.jsonl")