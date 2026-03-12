from neo4j import GraphDatabase
import networkx as nx

class GraphDBRetriever:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def retrieve(self, query_embedding, top_k=10):
        cypher = """
        MATCH (n:Chunk)
        RETURN n.id AS id, n.embedding AS embedding
        """
        with self.driver.session() as session:
            rows = session.run(cypher).data()

        # Compute cosine similarity manually
        scored = []
        for r in rows:
            emb = r["embedding"]
            score = float(query_embedding @ emb)
            scored.append((r["id"], score))

        scored = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
        node_ids = [s[0] for s in scored]

        # Build subgraph
        g = nx.Graph()
        with self.driver.session() as session:
            for nid in node_ids:
                rels = session.run("""
                    MATCH (n {id: $id})-[r]-(m)
                    RETURN n.id AS src, m.id AS dst
                """, id=nid).data()

                for r in rels:
                    g.add_edge(r["src"], r["dst"])

        return node_ids, g