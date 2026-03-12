import json
import numpy as np
import networkx as nx

class FileRetriever:
    def __init__(self, index_path):
        with open(index_path) as f:
            self.index = json.load(f)

    def retrieve(self, query_embedding, top_k=10):
        scored = []
        for node in self.index["nodes"]:
            emb = np.array(node["embedding"])
            score = float(query_embedding @ emb)
            scored.append((node["id"], score))

        scored = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
        node_ids = [s[0] for s in scored]

        # Build subgraph
        g = nx.Graph()
        for edge in self.index["edges"]:
            if edge["src"] in node_ids or edge["dst"] in node_ids:
                g.add_edge(edge["src"], edge["dst"])

        return node_ids, g