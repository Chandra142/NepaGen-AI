from __future__ import annotations

import hashlib
import json
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

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
VECTORSTORE_SCHEMA_VERSION = 1
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_SOURCE_TEXTS = 2000


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


def _legacy_store_exists() -> bool:
    return (VECTORSTORE_DIR / "index.pkl").exists() and INDEX_PATH.exists()


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

    source_texts = [
        str(text).strip()
        for text in dataframe["text"].dropna().tolist()
        if len(str(text).strip()) > 50
    ][:MAX_SOURCE_TEXTS]

    if not source_texts:
        raise ValueError("No usable source texts were found in the dataset.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = []
    metadatas = []
    for source_index, text in enumerate(source_texts):
        for chunk_index, chunk in enumerate(splitter.split_text(text)):
            chunks.append(chunk)
            metadatas.append(
                {
                    "source": "nepali_dataset.csv",
                    "source_index": source_index,
                    "chunk_index": chunk_index,
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
    embeddings = make_embeddings()
    manifest = _load_manifest()

    if _json_store_exists() and _manifest_is_current(manifest):
        try:
            return _load_json_store(embeddings)
        except Exception:
            pass

    if _legacy_store_exists():
        try:
            legacy_db = FAISS.load_local(
                str(VECTORSTORE_DIR),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            _save_json_store(
                legacy_db,
                source_text_count=(
                    manifest.get("source_text_count", len(legacy_db.index_to_docstore_id))
                    if manifest
                    else len(legacy_db.index_to_docstore_id)
                ),
                dataset_hash=_current_dataset_hash(),
            )
            return legacy_db
        except Exception:
            pass

    return build_vectorstore()