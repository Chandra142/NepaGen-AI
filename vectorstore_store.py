from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import time
from functools import lru_cache
from pathlib import Path

import faiss
import pandas as pd
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "nepali_dataset.csv"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
DOCSTORE_PATH = VECTORSTORE_DIR / "docstore.json"
MANIFEST_PATH = VECTORSTORE_DIR / "manifest.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTORSTORE_SCHEMA_VERSION = 2
CHUNK_SIZE = 250
CHUNK_OVERLAP = 64
MAX_SOURCE_TEXTS = 2000


@lru_cache(maxsize=1)
def make_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
    )


def _sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_dataset_hash() -> str | None:
    if not DATA_PATH.exists():
        return None
    return _sha256_file(DATA_PATH)


def _load_manifest() -> dict | None:
    if not MANIFEST_PATH.exists():
        return None

    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_manifest(*, document_count: int, source_text_count: int, dataset_hash: str | None) -> None:
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": VECTORSTORE_SCHEMA_VERSION,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "document_count": document_count,
        "source_text_count": source_text_count,
        "dataset_sha256": dataset_hash,
        "storage_format": "json-docstore",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_json_store(db: FAISS, *, source_text_count: int, dataset_hash: str | None) -> None:
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(db.index, str(INDEX_PATH))

    documents = []
    for index in sorted(db.index_to_docstore_id):
        docstore_id = db.index_to_docstore_id[index]
        document = db.docstore._dict[docstore_id]
        documents.append(
            {
                "id": docstore_id,
                "page_content": document.page_content,
                "metadata": document.metadata,
            }
        )

    DOCSTORE_PATH.write_text(
        json.dumps(
            {
                "schema_version": VECTORSTORE_SCHEMA_VERSION,
                "embedding_model": EMBEDDING_MODEL_NAME,
                "documents": documents,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    _write_manifest(
        document_count=len(documents),
        source_text_count=source_text_count,
        dataset_hash=dataset_hash,
    )


def _load_json_store(embeddings: HuggingFaceEmbeddings) -> FAISS:
    payload = json.loads(DOCSTORE_PATH.read_text(encoding="utf-8"))
    documents = payload.get("documents", [])

    if not documents:
        raise ValueError("The FAISS docstore JSON is empty.")

    docstore_mapping = {}
    index_to_docstore_id = {}

    for index, item in enumerate(documents):
        doc_id = item["id"]
        docstore_mapping[doc_id] = Document(
            page_content=item["page_content"],
            metadata=item.get("metadata") or {},
        )
        index_to_docstore_id[index] = doc_id

    index = faiss.read_index(str(INDEX_PATH))
    docstore = InMemoryDocstore(docstore_mapping)

    return FAISS(
        embeddings,
        index,
        docstore,
        index_to_docstore_id,
    )


def _json_store_exists() -> bool:
    return DOCSTORE_PATH.exists() and INDEX_PATH.exists() and MANIFEST_PATH.exists()


def _manifest_is_current(manifest: dict | None) -> bool:
    if not manifest:
        return False

    if manifest.get("schema_version") != VECTORSTORE_SCHEMA_VERSION:
        return False

    if manifest.get("embedding_model") != EMBEDDING_MODEL_NAME:
        return False

    current_hash = _current_dataset_hash()
    return manifest.get("dataset_sha256") == current_hash


def build_vectorstore() -> FAISS:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    dataframe = pd.read_csv(DATA_PATH)
    if "text" not in dataframe.columns:
        raise KeyError("Expected a 'text' column in nepali_dataset.csv.")

    def _clean_source_text(s: str) -> str:
        s = str(s).strip()
        s = re.sub(r"<!--.*?-->", " ", s, flags=re.S)
        s = re.sub(r"<[^>]+>", " ", s)
        s = unicodedata.normalize("NFKC", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    source_texts = []
    for text in dataframe["text"].dropna().tolist():
        cleaned = _clean_source_text(text)
        if len(cleaned) > 60:
            source_texts.append(cleaned)
        if len(source_texts) >= MAX_SOURCE_TEXTS:
            break

    if not source_texts:
        raise ValueError("No usable source texts were found in the dataset.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "?", "!", ",", " "]
    )

    def _is_mostly_devanagari(text: str) -> float:
        if not text:
            return 0.0
        devanagari = re.findall(r"[\u0900-\u097F]", text)
        return len(devanagari) / max(1, len(text))

    def _is_noise(text: str) -> bool:
        # obvious noise or spam patterns
        low = text.lower()
        if "read more" in low or "imagekhabar" in low or "--instant articles--" in low:
            return True
        if any(term in low for term in ["राजनीति", "government", "election", "congress", "bjp", "prime minister"]):
            # political/news junk often dominates irrelevant articles
            if len(text) < 220:
                return True
        if "http://" in low or "https://" in low or "www." in low:
            return True
        if "�" in text:
            return True
        if re.search(r"\{\{.*?\}\}", text):
            return True
        if re.search(r"(.)\1{7,}", text):
            return True
        if re.search(r"(\b\w+\b)(?:\s+\1){4,}", low):
            return True
        # extremely short
        if len(text) < 40:
            return True
        # English-heavy or product spec english blocks
        en_letters = re.findall(r"[A-Za-z]", text)
        if len(en_letters) / max(1, len(text)) > 0.6:
            return True
        return False

    def _quality_score(text: str) -> float:
        score = 0.0
        score += min(1.0, _is_mostly_devanagari(text) * 1.2)
        # prefer medium-length informative chunks
        l = len(text)
        if 120 <= l <= 800:
            score += 0.8
        elif l < 120:
            score += 0.2
        else:
            score += 0.5
        # penalize noise
        if _is_noise(text):
            score -= 1.5
        return round(max(0.0, score), 3)

    chunks = []
    metadatas = []
    seen_hashes = set()
    for source_index, text in enumerate(source_texts):
        for chunk_index, chunk in enumerate(splitter.split_text(text)):
            cleaned = _clean_source_text(chunk)
            if not cleaned:
                continue
            if _is_noise(cleaned):
                continue
            h = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            qscore = _quality_score(cleaned)
            # require a minimum quality
            if qscore <= 0.4:
                continue

            chunks.append(cleaned)
            metadatas.append(
                {
                    "source": "nepali_dataset.csv",
                    "source_index": source_index,
                    "chunk_index": chunk_index,
                    "quality_score": qscore,
                }
            )

    embeddings = make_embeddings()
    db = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas,
    )

    _save_json_store(
        db,
        source_text_count=len(source_texts),
        dataset_hash=_current_dataset_hash(),
    )

    return db


def load_vectorstore() -> FAISS:
    manifest = _load_manifest()

    if _json_store_exists() and _manifest_is_current(manifest):
        embedding_start = time.perf_counter()
        embeddings = make_embeddings()
        embedding_load_time = time.perf_counter() - embedding_start

        vectorstore_start = time.perf_counter()
        db = _load_json_store(embeddings)
        vectorstore_load_time = time.perf_counter() - vectorstore_start

        print(f"[startup] embedding load: {embedding_load_time:.2f}s")
        print(f"[startup] vectorstore load: {vectorstore_load_time:.2f}s")
        return db

    raise FileNotFoundError(
        "Vectorstore missing or invalid. Run ingest.py manually."
    )