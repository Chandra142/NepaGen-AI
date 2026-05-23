import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from vectorstore_store import build_vectorstore


if __name__ == "__main__":
    db = build_vectorstore()
    print(f"FAISS vector database created successfully with {len(db.index_to_docstore_id)} chunks!")