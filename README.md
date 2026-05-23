# NepaGen AI

NepaGen AI is a Streamlit-based Nepali RAG chatbot built with FAISS, LangChain, Groq, and HuggingFace embeddings.

## Project Structure

```text
NepaGen-AI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env
├── data/
│   └── nepali_dataset.csv
├── vectorstore/
│   ├── index.faiss
│   └── index.pkl
└── src/
    ├── ingest.py
    ├── rag.py
    └── retrieve.py
```

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your Groq API key to `.env`:

```env
GROQ_API_KEY=your_groq_api_key
```

## Run the App

```bash
streamlit run app.py
```

## Data and Vector Store

- `data/nepali_dataset.csv` contains the dataset used for ingestion.
- `vectorstore/` contains the FAISS index used by the chatbot.
- If you need to rebuild the index, run `src/ingest.py` from the project root.

## Deployment Notes

- Keep `app.py` at the repository root for Streamlit Cloud.
- Do not commit secrets such as `.env`.
- Keep `vectorstore/` if you want to deploy with a prebuilt FAISS index.

