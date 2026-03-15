import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import networkx as nx


# -----------------------------
# CONFIG: Allowed types
# -----------------------------
VALID_ENTITY_TYPES = {
    "Actor", "Authority", "AI_System", "Regulation",
    "Obligation", "Process", "Concept", "Document",
    "Article", "Chapter"
}

VALID_REL_TYPES = {
    "OBLIGATION", "COMPLIANCE", "GOVERNANCE", "SUPERVISION",
    "DEFINITION", "STRUCTURAL", "REFERENCE", "RELATED_TO"
}


# -----------------------------
# MAIN VALIDATOR
# -----------------------------
def validate_kg_jsonl(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSONL file not found: {path}")

    structural_errors = []
    semantic_errors = []
    G = nx.DiGraph()

    rel_count_dist = Counter()
    ent_count_dist = Counter()

    total_chunks = 0
    chunks_with_no_rel = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            total_chunks += 1

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                structural_errors.append(f"[Line {line_no}] Invalid JSON: {e}")
                continue

            # -----------------------------
            # Structural checks
            # -----------------------------
            for field in ["chunk_id", "source", "text", "entities", "relations"]:
                if field not in obj:
                    structural_errors.append(f"[Line {line_no}] Missing field: {field}")

            entities = obj.get("entities", [])
            relations = obj.get("relations", [])

            local_ids = set()
            for ent in entities:
                if not all(k in ent for k in ["local_id", "name", "type"]):
                    structural_errors.append(f"[Line {line_no}] Malformed entity: {ent}")
                    continue
                if ent["local_id"] in local_ids:
                    structural_errors.append(f"[Line {line_no}] Duplicate local_id: {ent['local_id']}")
                local_ids.add(ent["local_id"])

            for rel in relations:
                if not all(k in rel for k in ["source_local_id", "target_local_id", "type"]):
                    structural_errors.append(f"[Line {line_no}] Malformed relation: {rel}")
                    continue
                if rel["source_local_id"] not in local_ids:
                    structural_errors.append(f"[Line {line_no}] Relation source missing: {rel}")
                if rel["target_local_id"] not in local_ids:
                    structural_errors.append(f"[Line {line_no}] Relation target missing: {rel}")

            # -----------------------------
            # Semantic checks
            # -----------------------------
            for ent in entities:
                if ent["type"] not in VALID_ENTITY_TYPES:
                    semantic_errors.append(f"[Line {line_no}] Invalid entity type: {ent}")

            for rel in relations:
                if rel["type"] not in VALID_REL_TYPES:
                    semantic_errors.append(f"[Line {line_no}] Invalid relation type: {rel}")

            if not any(ent["type"] == "Regulation" for ent in entities):
                semantic_errors.append(f"[Line {line_no}] Missing regulation entity")

            if len(relations) == 0:
                chunks_with_no_rel += 1
                semantic_errors.append(f"[Line {line_no}] No relations extracted")

            if len(entities) == 0:
                semantic_errors.append(f"[Line {line_no}] No entities extracted")

            # -----------------------------
            # Graph-level accumulation
            # -----------------------------
            ent_count_dist[len(entities)] += 1
            rel_count_dist[len(relations)] += 1

            # Add to global graph
            for ent in entities:
                G.add_node(f"{obj['chunk_id']}::{ent['local_id']}", label=ent["name"], type=ent["type"])

            for rel in relations:
                src = f"{obj['chunk_id']}::{rel['source_local_id']}"
                tgt = f"{obj['chunk_id']}::{rel['target_local_id']}"
                G.add_edge(src, tgt, type=rel["type"])

    # -----------------------------
    # REPORT
    # -----------------------------
    print("\n==============================")
    print("STRUCTURAL VALIDATION RESULTS")
    print("==============================")
    if structural_errors:
        for e in structural_errors:
            print(" -", e)
    else:
        print("No structural issues detected.")

    print("\n==============================")
    print("SEMANTIC VALIDATION RESULTS")
    print("==============================")
    if semantic_errors:
        for e in semantic_errors:
            print(" -", e)
    else:
        print("No semantic issues detected.")

    print("\n==============================")
    print("GRAPH-LEVEL METRICS")
    print("==============================")
    print(f"Total chunks: {total_chunks}")
    print(f"Chunks with NO relations: {chunks_with_no_rel} ({chunks_with_no_rel/total_chunks:.2%})")

    print("\nRelation count distribution:")
    for k, v in sorted(rel_count_dist.items()):
        print(f"  {k} relations: {v} chunks")

    print("\nEntity count distribution:")
    for k, v in sorted(ent_count_dist.items()):
        print(f"  {k} entities: {v} chunks")

    # Graph connectivity
    components = list(nx.weakly_connected_components(G))
    print(f"\nWeakly connected components: {len(components)}")

    isolated = [n for n in G.nodes if G.degree(n) == 0]
    print(f"Isolated nodes: {len(isolated)}")

    print("\nDone.")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    # validate_kg_jsonl("./docs/reg_kg_triples.jsonl")
    # validate_kg_jsonl("./docs/reg_kg_triples_repaired.jsonl")
    validate_kg_jsonl("./docs/reg_kg_triples_rescued.jsonl")