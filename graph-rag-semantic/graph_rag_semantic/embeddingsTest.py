import os
import json
import re
import numpy as np
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.cluster import KMeans
from langchain_experimental.text_splitter import SemanticChunker
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from pypdf import PdfReader
from sklearn.cluster import KMeans

def test1(embedder):
    print(embedder.embed_query("hello world"))

def test2(embedder):
    docs = ["hello world", "this is a second test", "third segment"]
    print(embedder.embed_documents(docs))

def test3(embedder):
    splitter = SemanticChunker(embedder)
    docs = splitter.create_documents(["Hello world. This is a test. Another sentence."])
    print(docs)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    embedder = OllamaEmbeddings(model="mxbai-embed-large")
    print("\n=== 1. Minimal LangChain embedding test (no chunker) ===")
    try:
        test1(embedder)
    except Exception as e:
        print(f"error running test1: {e}")
   
    print("\n=== 2. Minimal LangChain batch test (exactly what SemanticChunker does) ===")
    try:
        test2(embedder)
    except Exception as e:
        print(f"error running test2: {e}")
  
    print("\n=== 3. If you want to test with SemanticChunker itself ===")
    try:
        test3(embedder)
    except Exception as e:
        print(f"error running test3: {e}")
