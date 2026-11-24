import csv
import hashlib
import io

from langchain_core.documents import Document
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="click")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="typer")

from langchain_text_splitters.character import RecursiveCharacterTextSplitter

from app.config import settings


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _hash(text: str) -> str:
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


def parse_pdf(data: bytes, source: str):
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        docs.append(Document(page_content=text, metadata={"source": source, "page": i + 1}))
    return docs


def parse_txt(data: bytes, source: str):
    text = data.decode("utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": source, "page": 1})]


def parse_md(data: bytes, source: str):
    return parse_txt(data, source)


def parse_csv(data: bytes, source: str):
    text = data.decode("utf-8", errors="ignore")
    reader = csv.reader(io.StringIO(text))
    rows = [" | ".join(r) for r in reader]
    return [Document(page_content="\n".join(rows), metadata={"source": source, "page": 1})]


PARSERS = {".pdf": parse_pdf, ".txt": parse_txt, ".md": parse_md, ".csv": parse_csv}


def parse_file(filename: str, data: bytes):
    import os

    ext = os.path.splitext(filename)[1].lower()
    if ext not in PARSERS:
        raise ValueError(f"Unsupported file type: {ext}")
    return PARSERS[ext](data, filename)


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    result = []
    for idx, c in enumerate(chunks):
        h = _hash(c.page_content)
        md = dict(c.metadata)
        md["chunk_index"] = idx
        md["chunk_hash"] = h
        result.append(Document(page_content=c.page_content, metadata=md))
    return result


def dedup_chunks(chunks):
    seen = set()
    out = []
    for c in chunks:
        h = c.metadata["chunk_hash"]
        if h in seen:
            continue
        seen.add(h)
        out.append(c)
    return out
