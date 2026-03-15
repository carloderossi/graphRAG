@echo off
title GraphRAG - Query

cd /d C:\Carlo\projects\graphRAG\graph-rag-graphdb
:: uv run .\graph_rag_graphdb\graphdb_client.py
uv run .\graph_rag_graphdb\multihop_obligations_client.py

pause




