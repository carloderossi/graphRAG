import time
from .logger import ComparisonLogger

class ComparisonRunner:
    def __init__(self, file_retriever, graphdb_retriever, llm):
        self.file = file_retriever
        self.db = graphdb_retriever
        self.llm = llm
        self.logger = ComparisonLogger()

    def run(self, query, query_embedding):
        results = {}

        # FILE
        t0 = time.time()
        f_nodes, f_graph = self.file.retrieve(query_embedding)
        f_answer = self.llm.answer(query, f_nodes)
        results["file"] = {
            "nodes": f_nodes,
            "graph": f_graph,
            "answer": f_answer,
            "latency": time.time() - t0
        }

        # GRAPHDB
        t0 = time.time()
        d_nodes, d_graph = self.db.retrieve(query_embedding)
        d_answer = self.llm.answer(query, d_nodes)
        results["db"] = {
            "nodes": d_nodes,
            "graph": d_graph,
            "answer": d_answer,
            "latency": time.time() - t0
        }

        self.logger.log(query, results)
        return results