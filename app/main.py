import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.ingest import parse_file, split_documents, dedup_chunks
from app.vector_store import vector_store
from app.retriever import hybrid_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)

llm = ChatOllama(
    model=settings.OLLAMA_MODEL,
    base_url=settings.OLLAMA_BASE_URL,
    num_ctx=settings.OLLAMA_NUM_CTX,
    temperature=0,
)

prompt = ChatPromptTemplate.from_template(
    "Answer based on context. Cite sources with [source].\nContext:\n{context}\nQuestion: {question}"
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    use_mmr: bool = False
    source_filter: str | None = None


class AskRequest(BaseModel):
    query: str
    top_k: int = 5
    source_filter: str | None = None


class ChunkOut(BaseModel):
    content: str
    score: float
    metadata: dict


class SearchResponse(BaseModel):
    chunks: list[ChunkOut]
    retrieval_latency_ms: float


class IngestResponse(BaseModel):
    filename: str
    total_chunks: int
    new_chunks_added: int
    duplicates_skipped: int
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    start = time.time()
    data = await file.read()
    docs = parse_file(file.filename, data)
    chunks = split_documents(docs)
    total = len(chunks)
    chunks = dedup_chunks(chunks)
    new, skipped = vector_store.add_documents(chunks)
    latency = (time.time() - start) * 1000
    return IngestResponse(
        filename=file.filename,
        total_chunks=total,
        new_chunks_added=new,
        duplicates_skipped=skipped + (total - len(chunks)),
        latency_ms=latency,
    )


@app.get("/sources")
def sources():
    return vector_store.list_sources()


@app.delete("/sources/{source_name}")
def delete_source(source_name: str):
    n = vector_store.delete_by_source(source_name)
    return {"deleted": n}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    start = time.time()
    results = hybrid_retriever.retrieve(req.query, top_k=req.top_k, use_mmr=req.use_mmr, source_filter=req.source_filter)
    chunks = [ChunkOut(content=r["document"].page_content, score=r["score"], metadata=r["metadata"]) for r in results]
    latency = (time.time() - start) * 1000
    return SearchResponse(chunks=chunks, retrieval_latency_ms=latency)


@app.post("/ask")
def ask(req: AskRequest):
    start = time.time()
    results = hybrid_retriever.retrieve(req.query, top_k=req.top_k, source_filter=req.source_filter)
    if not results or results[0]["score"] < settings.SIMILARITY_THRESHOLD:
        return {"answer": "Insufficient context to answer.", "citations": [], "latency_ms": (time.time() - start) * 1000}
    context = "\n\n".join(r["document"].page_content for r in results)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": req.query})
    return {"answer": answer, "citations": [r["metadata"] for r in results], "latency_ms": (time.time() - start) * 1000}
