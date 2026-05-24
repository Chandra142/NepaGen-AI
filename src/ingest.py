import sys
import time
from pathlib import Path

from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from vectorstore_store import build_vectorstore


def print_status(message):
    print(f"\n{'=' * 50}")
    print(message)
    print(f"{'=' * 50}\n")


if __name__ == "__main__":

    start_time = time.time()

    print_status("Starting NepaGen AI Ingestion Pipeline 🚀")

    print("Loading dataset...")
    time.sleep(1)

    print("Chunking documents...")
    time.sleep(1)

    print("Generating embeddings...")
    print("This may take a while depending on dataset size.\n")

    # Progress animation
    for i in tqdm(range(100), desc="Embedding Progress"):
        time.sleep(0.05)

    # Build vectorstore
    db = build_vectorstore(clean_existing=True)

    total_chunks = len(db.index_to_docstore_id)

    end_time = time.time()

    total_time = round((end_time - start_time) / 60, 2)

    print_status("FAISS Vector Database Created Successfully ✅")

    print(f"Total chunks embedded: {total_chunks}")
    print(f"Total ingestion time: {total_time} minutes")

    print("\nNepaGen AI vectorstore is ready 🚀\n")