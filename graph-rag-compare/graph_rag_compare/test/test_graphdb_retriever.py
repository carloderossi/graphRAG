from graph_rag_compare.retrievers.graphdb_retriever import GraphDBRetriever
from graph_rag_compare.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
import numpy as np

def test_graphdb_retriever():
    retriever = GraphDBRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    fake_query_emb = np.random.rand(1024)
    nodes, graph = retriever.retrieve(fake_query_emb, top_k=5)

    print("Retrieved nodes:", nodes)
    print("Graph nodes:", graph.nodes())
    print("Graph edges:", graph.edges())

    retriever.close()