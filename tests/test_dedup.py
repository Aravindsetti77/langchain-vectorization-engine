import hashlib

from app.ingest import split_documents, dedup_chunks, parse_txt


def test_dedup():
    docs = parse_txt(b"hello world " * 50, "a.txt")
    chunks = split_documents(docs)
    duped = chunks + chunks
    out = dedup_chunks(duped)
    assert len(out) == len(chunks)
    hashes = [c.metadata["chunk_hash"] for c in out]
    assert len(set(hashes)) == len(hashes)
