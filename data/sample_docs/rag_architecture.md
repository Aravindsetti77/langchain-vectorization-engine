Retrieval-Augmented Generation Architecture

RAG combines dense vector retrieval with large language models to ground answers in external knowledge. Documents are chunked, embedded with SentenceTransformers, and stored in a vector database such as ChromaDB using HNSW indexing with cosine similarity.

Hybrid retrieval fuses dense semantic search with sparse BM25 lexical search via Reciprocal Rank Fusion. This captures both semantic intent and exact keyword matches for acronyms and identifiers. Metadata filtering allows scoping retrieval to specific sources.

Generation uses a prompt template that injects retrieved context and instructs the model to cite sources. A similarity threshold guards against hallucination when context is insufficient. LCEL chains compose prompt, LLM, and output parser cleanly.

Evaluation uses Recall@k, Mean Reciprocal Rank, and latency percentiles. Deduplication via content hashing ensures idempotent ingestion and avoids duplicate vectors on re-upload.
