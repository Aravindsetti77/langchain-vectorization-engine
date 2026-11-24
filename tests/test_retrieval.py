import tempfile
import os

from langchain_core.embeddings import FakeEmbeddings
from langchain_core.documents import Document

import app.vector_store as vs_mod
from app.vector_store import VectorStore
from app.ingest import parse_txt, split_documents, dedup_chunks
from app import retriever as ret_mod
from app.retriever import HybridRetriever


def test_hybrid_retrieval():
    tmp = tempfile.mkdtemp()
    os.environ["CHROMA_PERSIST_DIR"] = tmp
    from app.config import settings

    settings.CHROMA_PERSIST_DIR = tmp
    vs = VectorStore(embeddings=FakeEmbeddings(size=384))
    vs_mod.vector_store = vs
    ret = HybridRetriever()
    ret_mod.vector_store = vs
    ret_mod.hybrid_retriever = ret

    docs = parse_txt(b"LangChain is a framework for LLM apps.", "t.txt")
    chunks = dedup_chunks(split_documents(docs))
    vs.add_documents(chunks)
    res = ret.retrieve("LangChain framework", top_k=3)
    assert len(res) > 0
    assert "score" in res[0]
