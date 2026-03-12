from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text):
    return model.encode([text])[0]

def evaluate(file_res, db_res):
    return {
        "answer_similarity": float(cosine_similarity(
            [embed(file_res["answer"])],
            [embed(db_res["answer"])]
        )[0][0]),
        "node_overlap": len(set(file_res["nodes"]) & set(db_res["nodes"])) /
                        max(len(file_res["nodes"]), len(db_res["nodes"])),
        "latency_ratio": file_res["latency"] / db_res["latency"]
    }