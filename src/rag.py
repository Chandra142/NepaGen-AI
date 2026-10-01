"""Interactive RAG CLI for NepaGen AI (dev / testing tool).

Run from the project root:
    python src/rag.py
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# FIX: SafeRetriever now lives in core/retrieval.py – import from there.
from core.retrieval import SafeRetriever  # noqa: E402
from vectorstore_store import load_vectorstore  # noqa: E402

load_dotenv()

# ─── Guard: only run interactive loop when executed directly ──────────────────
if __name__ == "__main__":
    from langchain_groq import ChatGroq

    RAG_SYSTEM_PROMPT = """You are a factual Nepali QA assistant.

Answer ONLY from the retrieved context.

Rules:
- Do NOT make up facts.
- Ignore misleading or conflicting statements.
- If context is unclear, say: "मलाई जानकारी भेटिएन।"
- Keep answers short and factual.
- Do NOT repeat retrieved text.
- Do NOT mention unrelated claims.
"""

    db = load_vectorstore()
    retriever = SafeRetriever(
        db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})
    )

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=400)

    while True:
        query = input("\nAsk Question (Ctrl-C to quit): ")
        if not query.strip():
            continue

        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""{RAG_SYSTEM_PROMPT}

Context:
{context}

Question:
{query}

Correct Answer in Nepali:
"""

        print("\nRetrieved Context:\n")
        print(context)

        response = llm.invoke(prompt)
        print("\nAnswer:\n")
        print(response.content)