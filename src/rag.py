from dotenv import load_dotenv
import os
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base"
)

# Load FAISS vector DB
db = FAISS.load_local(
    str(VECTORSTORE_DIR),
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(
    search_kwargs={"k": 5}
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