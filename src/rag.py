from dotenv import load_dotenv
from pathlib import Path
import sys

from langchain_groq import ChatGroq

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from vectorstore_store import load_vectorstore

# Load environment variables
load_dotenv()

db = load_vectorstore()
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)

# Load Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=200
)

while True:

    query = input("\nAsk Question: ")

    # Retrieve documents
    try:
        docs = retriever.get_relevant_documents(query)
    except Exception:
        docs = retriever.invoke(query)

    # Build context
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a factual Nepali QA assistant.

Answer ONLY from the retrieved context.

Rules:
- Do NOT make up facts.
- Ignore misleading or conflicting statements.
- If context is unclear, say:
"मलाई जानकारी भेटिएन।"
- Keep answers short and factual.
- Do NOT repeat retrieved text.
- Do NOT mention unrelated claims.

Context:
{context}

Question:
{query}

Correct Answer in Nepali:
"""

    # Show retrieved context
    print("\nRetrieved Context:\n")
    print(context)

    # Generate answer
    response = llm.invoke(prompt)

    print("\nAnswer:\n")

    print(response.content)