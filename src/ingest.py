import pandas as pd
from tqdm import tqdm
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "nepali_dataset.csv"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Use smaller subset for testing
texts = [
    text for text in df["text"].dropna().tolist()
    if len(text) > 50
][:2000]

print(f"Loaded {len(texts)} texts")

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

documents = splitter.create_documents(texts)

print(f"Created {len(documents)} chunks")

# Load embedding model
model = SentenceTransformer(
    "intfloat/multilingual-e5-base"
)

# Generate embeddings manually
doc_texts = [doc.page_content for doc in documents]

print("Generating embeddings...")

embeddings = model.encode(
    doc_texts,
    show_progress_bar=True,
    batch_size=32
)

print("Embeddings generated!")

# Create FAISS DB
db = FAISS.from_embeddings(
    text_embeddings=list(zip(doc_texts, embeddings)),
    embedding=model
)

# Save vector DB
db.save_local(str(VECTORSTORE_DIR))

print("FAISS vector database created successfully!")