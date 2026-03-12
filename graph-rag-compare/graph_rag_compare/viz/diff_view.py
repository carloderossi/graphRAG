from typing import Iterable, Dict, Any

def compute_set_diff(a: Iterable[Any], b: Iterable[Any]) -> Dict[str, list]:
    a_set = set(a)
    b_set = set(b)
    return {
        "only_in_a": list(a_set - b_set),
        "only_in_b": list(b_set - a_set),
        "intersection": list(a_set & b_set),
    }

def node_diff(file_nodes, db_nodes):
    return compute_set_diff(file_nodes, db_nodes)

def chunk_diff(file_chunks, db_chunks):
    return compute_set_diff(file_chunks, db_chunks)