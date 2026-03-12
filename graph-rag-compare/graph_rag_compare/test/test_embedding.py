from graph_rag_compare.embedding import embed_query

def test_embedding():
    emb = embed_query("What is AI governance?")
    print("Embedding shape:", emb.shape)