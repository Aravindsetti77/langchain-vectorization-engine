# LangChain Vectorization & Hybrid RAG Microservice

100% locally runnable FastAPI microservice for document vectorization and hybrid RAG using ChromaDB (HNSW cosine), `BAAI/bge-small-en-v1.5`, BM25 + dense Reciprocal Rank Fusion, and local Ollama `llama3.2` with grounded citations.

## System Architecture

```
Ingest: pypdf/text/CSV -> RecursiveCharacterTextSplitter -> SHA-256 dedup -> ChromaDB
Retrieval: Dense (0.65) + BM25 (0.35) RRF (+MMR) -> threshold gate
Generation: LCEL prompt | ChatOllama(num_ctx=8192) | StrOutputParser
```

## Key Design Decisions

| Decision | Why |
|---|---|
| SHA-256 chunk hashing | Idempotent ingestion, no duplicate vectors |
| BAAI/bge-small-en-v1.5 384-dim normalized | CPU/edge efficient |
| 0.65 dense / 0.35 BM25 RRF | Semantic intent + exact acronyms |
| SIMILARITY_THRESHOLD=0.35 | Guardrail vs hallucination |
| num_ctx=8192 | Prevent silent truncation |

## Project Structure

```
app/config.py - settings
app/ingest.py - parsers + splitter + dedup
app/vector_store.py - Chroma + embeddings
app/retriever.py - hybrid RRF
app/main.py - FastAPI endpoints
eval/eval_retrieval.py - benchmark
```

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
ollama pull llama3.2
uvicorn app.main:app --reload
```

Docker:

```bash
docker compose up --build
```

## API

- POST /ingest (multipart file)
- GET /sources
- DELETE /sources/{source_name}
- POST /search {query,top_k,use_mmr,source_filter}
- POST /ask {query,top_k,source_filter}
- GET /health

## Evaluation

```bash
pytest -W error::DeprecationWarning
python eval/eval_retrieval.py
```

| Recall@3 | MRR | p95 Latency (ms) |
|---|---|---|
| 0.00 | 0.00 | 20.10 |
