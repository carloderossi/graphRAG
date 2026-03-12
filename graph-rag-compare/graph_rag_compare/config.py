from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

# Paths
DOCS_DIR = BASE_DIR.parent / "docs"
SEMANTIC_INDEX_PATH = BASE_DIR.parent / "graph-rag-semantic" / "ai_reg_semantic_index.json"
LOG_PATH = BASE_DIR / "graph_rag_compare" / "comparison_log.jsonl"

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Retrieval
TOP_K_NODES = int(os.getenv("TOP_K_NODES", "10"))

# Models
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "mxbai-embed-large:latest")