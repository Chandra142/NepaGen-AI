# NepaGen AI

NepaGen AI is a premium Nepali Streamlit chat app powered by FAISS, LangChain, Groq, and HuggingFace embeddings.

Live app: [rag142.streamlit.app](https://rag142.streamlit.app/)

Dataset: [Sakonii/nepalitext-language-model-dataset](https://huggingface.co/datasets/Sakonii/nepalitext-language-model-dataset)

## Overview

NepaGen AI provides grounded RAG responses over a Nepali dataset with a polished chat experience, persistent conversation history, and responsive UI behavior on Streamlit Cloud.

## Features

- Nepali-language RAG chat
- FAISS-backed retrieval
- Groq-powered generation
- MMR retrieval for better context selection
- Persistent chat history
- Premium dark UI with responsive layout

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── nepali_dataset.csv
├── vectorstore/
│   ├── index.faiss
│   ├── docstore.json
│   └── manifest.json
└── src/
    ├── ingest.py
    ├── rag.py
    └── retrieve.py
```

## Local Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your Groq API key in a local `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

## Run Locally

```bash
streamlit run app.py
```

## Rebuild the Vectorstore

If you update the dataset or embedding model, rebuild the FAISS store from the project root:

```bash
python src/ingest.py
```

This regenerates the vectorstore files in `vectorstore/`.

## Deployment Notes

- Keep `app.py` at the repository root for Streamlit Cloud.
- Commit the `vectorstore/` directory if you want deployment to use the prebuilt index.
- Do not commit secrets such as `.env`.
- If retrieval fails after an embedding change, rebuild the vectorstore before redeploying.

## Data Source

The app uses the Nepali text dataset from Hugging Face:

- [Sakonii/nepalitext-language-model-dataset](https://huggingface.co/datasets/Sakonii/nepalitext-language-model-dataset)
