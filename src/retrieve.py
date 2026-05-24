import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from vectorstore_store import load_vectorstore


class SafeRetriever:
    def __init__(self, retriever):
        self._retriever = retriever

    def get_relevant_documents(self, query):
        try:
            return self._retriever.get_relevant_documents(query)
        except Exception as exc:
            print("[retrieve] retriever.get_relevant_documents failed:", repr(exc))
            return []

    def invoke(self, query):
        try:
            return self._retriever.invoke(query)
        except Exception as exc:
            print("[retrieve] retriever.invoke failed:", repr(exc))
            return []

db = load_vectorstore()

# Create retriever
retriever = SafeRetriever(
    db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 12},
    )
)

while True:

    query = input("\nAsk Question: ")

    docs = retriever.get_relevant_documents(query)

    print("\nRetrieved Documents:\n")

    for i, doc in enumerate(docs):

        print(f"\nDocument {i+1}:\n")

        print(doc.page_content)