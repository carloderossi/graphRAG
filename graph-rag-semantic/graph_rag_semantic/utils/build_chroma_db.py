import json
import chromadb
from chromadb.config import Settings
from config import SEMANTIC_INDEX_PATH, get_chroma_db_path

def load_chunks():
    with open(SEMANTIC_INDEX_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = []
    for chunk_id, obj in data["chunks"].items():
        chunks.append({
            "chunk_id": chunk_id,
            "text": obj["text"],
            "source": obj.get("source", ""),
            "embedding": obj["vec"]
        })
    return chunks

def build_chroma():
    print("Loading chunks...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.")

    chroma_db_path = get_chroma_db_path()
    print(f"Targeting Chroma DB path '{chroma_db_path}'...")
    client = chromadb.PersistentClient(path=chroma_db_path)
    print("Getting collection...")
    collection = client.get_or_create_collection("reg_chunks")

    ids, docs, embs, metas = [], [], [], []

    print("Storing chunks in Chroma Vector DB...")
    for i, item in enumerate(chunks):
        ids.append(str(i))
        docs.append(item["text"])
        embs.append(item["embedding"])
        metas.append({
            "chunk_id": item["chunk_id"],
            "source": item["source"]
        })

    collection.add(
        ids=ids,
        documents=docs,
        embeddings=embs,
        metadatas=metas
    )

    print(f"Succesfully built Chroma DB with {len(ids)} chunks.")

if __name__ == "__main__":
    print("Building Chroma DB...")
    build_chroma()