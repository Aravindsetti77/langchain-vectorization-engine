import time
import os

from app.ingest import parse_file, split_documents, dedup_chunks
from app.vector_store import vector_store
from app.retriever import hybrid_retriever


def main():
    d = "data/sample_docs"
    for f in os.listdir(d):
        data = open(os.path.join(d, f), "rb").read()
        docs = parse_file(f, data)
        chunks = dedup_chunks(split_documents(docs))
        vector_store.add_documents(chunks)

    questions = [
        ("What is LangChain?", "doc1.txt"),
    ] * 10

    latencies = []
    recalls = []
    mrrs = []
    for q, src in questions:
        t = time.time()
        res = hybrid_retriever.retrieve(q, top_k=3)
        latencies.append((time.time() - t) * 1000)
        hit = any(r["metadata"].get("source") == src for r in res)
        recalls.append(1.0 if hit else 0.0)
        rr = 0.0
        for i, r in enumerate(res):
            if r["metadata"].get("source") == src:
                rr = 1.0 / (i + 1)
                break
        mrrs.append(rr)

    import statistics
    p95 = sorted(latencies)[int(0.95 * (len(latencies) - 1))]
    print(f"Recall@3: {sum(recalls)/len(recalls):.3f}")
    print(f"MRR: {sum(mrrs)/len(mrrs):.3f}")
    print(f"p95_latency_ms: {p95:.2f}")


if __name__ == "__main__":
    main()
