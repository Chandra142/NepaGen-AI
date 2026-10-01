"""Unified retriever wrappers for NepaGen AI.

Previously SafeRetriever was duplicated in app.py, src/rag.py, and src/retrieve.py.
This is the single source of truth.
"""
from __future__ import annotations


class _EmptyRetriever:
    """No-op retriever used when the vectorstore is unavailable."""

    def get_relevant_documents(self, query: str) -> list:  # noqa: ARG002
        return []

    def invoke(self, query: str) -> list:  # noqa: ARG002
        return []


class SafeRetriever:
    """Wraps any LangChain retriever with exception-safe methods."""

    def __init__(self, retriever_obj) -> None:
        self._retriever = retriever_obj

    def get_relevant_documents(self, query: str) -> list:
        try:
            return self._retriever.get_relevant_documents(query)
        except Exception as exc:
            print(f"[retrieval] get_relevant_documents failed: {exc!r}")
            return []

    def invoke(self, query: str) -> list:
        try:
            return self._retriever.invoke(query)
        except Exception as exc:
            print(f"[retrieval] invoke failed: {exc!r}")
            return []


def make_retriever(
    db,
    *,
    k: int = 5,
    fetch_k: int = 20,
) -> SafeRetriever | _EmptyRetriever:
    """Return a SafeRetriever for *db*, or an _EmptyRetriever if the index is empty."""
    if getattr(db.index, "ntotal", 0) > 0:
        return SafeRetriever(
            db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": fetch_k},
            )
        )
    print("[retrieval] empty vectorstore — using no-op retriever")
    return _EmptyRetriever()


def safe_retrieve(retriever, query: str) -> list:
    """Convenience wrapper around retriever.invoke()."""
    return retriever.invoke(query)
