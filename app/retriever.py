from langchain_community.retrievers import BM25Retriever

from langchain_core.documents import Document

from app.vector_store import vector_store
from app.config import settings


def _rrf(results_list, k=60):
    scores = {}
    docs_map = {}
    for results in results_list:
        for rank, doc in enumerate(results):
            key = doc.metadata.get("chunk_hash") or doc.page_content
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            docs_map[key] = doc
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(docs_map[k], v) for k, v in ranked]


class HybridRetriever:
    def __init__(self):
        self.weights = (0.65, 0.35)

    def _dense(self, query, top_k, use_mmr, source_filter):
        vs = vector_store.store
        kwargs = {}
        if source_filter:
            kwargs["filter"] = {"source": source_filter}
        if use_mmr:
            docs = vs.max_marginal_relevance_search(query, k=top_k, fetch_k=max(top_k * 4, 20), **kwargs)
        else:
            docs = vs.similarity_search(query, k=top_k, **kwargs)
        return docs

    def _sparse(self, query, top_k, source_filter):
        all_docs = vector_store.get_all_documents()
        if source_filter:
            all_docs = [d for d in all_docs if d.metadata.get("source") == source_filter]
        if not all_docs:
            return []
        bm25 = BM25Retriever.from_documents(all_docs)
        bm25.k = top_k
        return bm25.invoke(query)

    def retrieve(self, query, top_k=5, use_mmr=False, source_filter=None):
        dense_docs = self._dense(query, top_k, use_mmr, source_filter)
        sparse_docs = self._sparse(query, top_k, source_filter)
        fused = _rrf([dense_docs, sparse_docs])
        out = []
        for doc, score in fused[:top_k]:
            out.append({"document": doc, "score": float(score), "metadata": doc.metadata})
        # normalize scores to 0-1
        if out:
            mx = max(o["score"] for o in out) or 1.0
            for o in out:
                o["score"] = o["score"] / mx
        return out


hybrid_retriever = HybridRetriever()
