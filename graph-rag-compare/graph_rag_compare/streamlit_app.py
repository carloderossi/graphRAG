import streamlit as st
from viz.pyvis_graph import render_graph
from harness.comparison_runner import ComparisonRunner
from harness.metrics import evaluate
from retrievers.file_retriever import FileRetriever
from retrievers.graphdb_retriever import GraphDBRetriever

from viz.diff_view import node_diff, chunk_diff

st.set_page_config(layout="wide")

st.title("GraphRAG Comparison — File vs GraphDB")

query = st.text_input("Enter your question")

if st.button("Run"):
    file = FileRetriever("../graph-rag-semantic/ai_reg_semantic_index.json")
    db = GraphDBRetriever("bolt://localhost:7687", "neo4j", "password123")

    runner = ComparisonRunner(file, db, llm=None)  # plug your LLM wrapper
    results = runner.run(query, query_embedding=None)  # plug embedding

    file_res = results["file"]
    db_res = results["db"]

    st.subheader("Answers")
    col1, col2 = st.columns(2)
    col1.write(file_res["answer"])
    col2.write(db_res["answer"])

    st.subheader("Metrics")
    st.json(evaluate(file_res, db_res))

    st.subheader("Graphs")
    col1, col2 = st.columns(2)
    col1.components.v1.html(render_graph(file_res["graph"]), height=500)
    col2.components.v1.html(render_graph(db_res["graph"]), height=500)

import subprocess
import sys
from pathlib import Path

def main():
    """
    Wrapper so `graph-rag-compare-app` launches Streamlit.
    """
    app_path = Path(__file__).resolve()
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])