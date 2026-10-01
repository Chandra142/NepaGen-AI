"""NepaGen AI – vectorstore ingestion pipeline.

Run from the project root:
    python src/ingest.py
"""
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from vectorstore_store import build_vectorstore  # noqa: E402


def _section(msg: str) -> None:
    bar = "=" * 54
    print(f"\n{bar}\n  {msg}\n{bar}\n")


if __name__ == "__main__":
    start = time.time()

    _section("NepaGen AI — Ingestion Pipeline 🚀")

    print("Step 1/3 — Loading & cleaning dataset …")
    print("Step 2/3 — Chunking and filtering documents …")
    print("Step 3/3 — Generating embeddings and building FAISS index …")
    print("          (this may take several minutes depending on dataset size)\n")

    # FIX: pass clean_existing=True so old files are removed before rebuild.
    # FIX: the fake tqdm sleep loop has been removed; build_vectorstore() does
    #      real work here and progress is visible via console output.
    db = build_vectorstore(clean_existing=True)

    total_chunks = len(db.index_to_docstore_id)
    elapsed = round((time.time() - start) / 60, 2)

    _section("FAISS Vector Database Created Successfully ✅")
    print(f"  Total chunks embedded : {total_chunks}")
    print(f"  Total ingestion time  : {elapsed} minutes")
    print("\nNepaGen AI vectorstore is ready 🚀\n")