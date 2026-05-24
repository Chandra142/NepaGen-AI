from datasets import load_dataset
import pandas as pd
import os

# ====================================
# SETTINGS
# ====================================

DATASET_SIZE = 25000
MIN_TEXT_LENGTH = 50

OUTPUT_PATH = "data/nepali_dataset.csv"

# ====================================
# LOAD DATASET IN STREAMING MODE
# ====================================

print("\nLoading dataset in streaming mode...\n")

dataset = load_dataset(
    "Sakonii/nepalitext-language-model-dataset",
    split="train",
    streaming=True
)

# ====================================
# COLLECT DATA
# ====================================

texts = []

print(f"Collecting {DATASET_SIZE} rows...\n")

for i, sample in enumerate(dataset):

    text = sample.get("text", "")

    if (
        text
        and isinstance(text, str)
        and len(text) > MIN_TEXT_LENGTH
    ):
        texts.append(text)

    # Progress log
    if i % 1000 == 0:
        print(f"Processed: {i}")

    # Stop after desired size
    if len(texts) >= DATASET_SIZE:
        break

# ====================================
# CREATE DATAFRAME
# ====================================

df = pd.DataFrame({
    "text": texts
})

# ====================================
# REMOVE DUPLICATES
# ====================================

before = len(df)

df = df.drop_duplicates(subset=["text"])

after = len(df)

print(f"\nRemoved duplicates: {before - after}")

# ====================================
# CREATE DATA FOLDER
# ====================================

os.makedirs("data", exist_ok=True)

# ====================================
# SAVE CSV
# ====================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n===================================")
print("Dataset saved successfully!")
print(f"Rows saved: {len(df)}")
print(f"Saved at: {OUTPUT_PATH}")
print("===================================\n")