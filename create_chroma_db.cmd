@echo off
title GraphRAG - Create Chroma Vector DB

cd /d C:\Carlo\projects\graphRAG\graph-rag-semantic
uv run .\graph_rag_semantic\utils\build_chroma_db.py

pause




