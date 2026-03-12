from graph_rag_compare.retrievers.file_retriever import FileRetriever
from graph_rag_compare.config import SEMANTIC_INDEX_PATH
import numpy as np

def test_file_retriever():
    retriever = FileRetriever(SEMANTIC_INDEX_PATH)
    fake_query_emb = np.random.rand(1024)  # mxbai-embed-large dimension
    nodes, graph = retriever.retrieve(fake_query_emb, top_k=5)

    print("Retrieved nodes:", nodes)
    print("Graph nodes:", graph.nodes())
    print("Graph edges:", graph.edges())