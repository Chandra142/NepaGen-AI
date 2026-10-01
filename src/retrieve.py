"""Interactive retrieval CLI for NepaGen AI (dev / testing tool).

Run from the project root:
    python src/retrieve.py
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# FIX: SafeRetriever now lives in core/retrieval.py – import from there.
from core.retrieval import SafeRetriever  # noqa: E402
from vectorstore_store import load_vectorstore  # noqa: E402

# ─── Guard: only run interactive loop when executed directly ──────────────────
if __name__ == "__main__":
    db = load_vectorstore()

    retriever = SafeRetriever(
        db.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 12})
    )

    while True:
        query = input("\nAsk Question (Ctrl-C to quit): ")
        if not query.strip():
            continue

        docs = retriever.get_relevant_documents(query)

        print("\nRetrieved Documents:\n")
        for i, doc in enumerate(docs):
            print(f"\nDocument {i + 1}:\n")
            print(doc.page_content)