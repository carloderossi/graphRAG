"""
+---docs
+---graph-rag-compare
ª   +---graph_rag_compare
ª       +---harness
ª       +---notebooks
ª       +---retrievers
ª       +---test
ª       +---viz
+---graph-rag-semantic
ª   +---graph_rag_semantic
ª       +---chroma
ª       +---test
ª       +---utils        <-----
"""
import os
from pathlib import Path

EMBED_MODEL = "mxbai-embed-large:latest"
LLM_MODEL = "llama3.1:8b"

current_file = Path(__file__).resolve()

def get_project_root():
    # root = current_file.parents[0]   # same as .parent
    # grandparent = current_file.parents[1]
    # project_root = current_file.parents[2]
    return current_file.parents[3]

def get_abs_config_path(cfg_path):
    """Return an absolute path for the given configuration file.

    Parameters
    ----------
    cfg_path : str
        Relative or absolute path to the configuration file.

    Returns
    -------
    str
        Absolute path string.  Prints the path or an error message.
    """
    p = Path(cfg_path).resolve()

    if p.exists():
        abs_path = str(p)
        print("Absolute path:", abs_path)
    else:
        print("File does not exist:", p)
    return abs_path

def get_chroma_db_path():
    return current_file.parents[1] / "chroma"

def get_docs_folder():
    return current_file.parents[3] / "docs"

SEMANTIC_INDEX_PATH = get_docs_folder() / "ai_reg_semantic_index.json"
KG_PATH = get_docs_folder() / "reg_kg_triples_repaired.jsonl"
CHROMA_PATH = get_chroma_db_path()
