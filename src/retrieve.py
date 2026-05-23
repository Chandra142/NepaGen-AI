import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from vectorstore_store import load_vectorstore

db = load_vectorstore()

# Create retriever
retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

while True:

    query = input("\nAsk Question: ")

    docs = retriever.invoke(query)

    print("\nRetrieved Documents:\n")

    for i, doc in enumerate(docs):

        print(f"\nDocument {i+1}:\n")

        print(doc.page_content)