from langchain_chroma import Chroma

from app.config import settings


class VectorStore:
    def __init__(self, embeddings=None):
        self._embeddings = embeddings
        self._store = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings

            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    @property
    def store(self):
        if self._store is None:
            self._store = Chroma(
                persist_directory=settings.CHROMA_PERSIST_DIR,
                embedding_function=self.embeddings,
                collection_metadata={"hnsw:space": "cosine"},
            )
        return self._store

    def add_documents(self, docs):
        existing = set()
        try:
            data = self.store.get()
            for m in data.get("metadatas", []) or []:
                if m and "chunk_hash" in m:
                    existing.add(m["chunk_hash"])
        except Exception:
            pass
        new_docs = [d for d in docs if d.metadata.get("chunk_hash") not in existing]
        skipped = len(docs) - len(new_docs)
        if new_docs:
            ids = [d.metadata["chunk_hash"] for d in new_docs]
            self.store.add_documents(documents=new_docs, ids=ids)
        return len(new_docs), skipped

    def list_sources(self):
        data = self.store.get()
        counts = {}
        for m in data.get("metadatas", []) or []:
            src = m.get("source") if m else None
            if src is not None:
                counts[src] = counts.get(src, 0) + 1
        return [{"source": k, "chunks": v} for k, v in counts.items()]

    def delete_by_source(self, source_name):
        data = self.store.get(where={"source": source_name})
        ids = data.get("ids", []) or []
        if ids:
            self.store.delete(ids=ids)
        return len(ids)

    def get_all_documents(self):
        data = self.store.get()
        docs = []
        for i, content in enumerate(data.get("documents", []) or []):
            md = (data.get("metadatas", []) or [])[i] if i < len(data.get("metadatas", []) or []) else {}
            from langchain_core.documents import Document

            docs.append(Document(page_content=content, metadata=md or {}))
        return docs


vector_store = VectorStore()
