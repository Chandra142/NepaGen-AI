from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path

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