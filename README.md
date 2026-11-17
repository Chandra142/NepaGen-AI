# NepaGen AI

NepaGen AI is a Nepali-language RAG chat app built with Streamlit, FAISS, LangChain, Groq, and HuggingFace embeddings. It answers questions in Nepali (and English) grounded in a prebuilt vector store, with persistent conversation history and a responsive dark UI that works on phones, tablets, and desktop.

Live app: [rag142.streamlit.app](https://rag142.streamlit.app/)

Dataset: [Sakonii/nepalitext-language-model-dataset](https://huggingface.co/datasets/Sakonii/nepalitext-language-model-dataset)

## Features

- **Grounded RAG answers** in Nepali, with retrieval-backed responses and expandable **References** under each answer
- **Hybrid retrieval**: vector search (FAISS + MMR) fused with a lightweight lexical index — normalization of common Nepali/English spelling variants, answer-sentence scoring, small-number matching, and one complete window per source (neighboring chunks are merged so answers are not cut off mid-sentence)
- **Follow-up awareness**: pronoun-style follow-ups (“त्यसको …”) resolve against the previous user question
- **Query routing**: greetings, small talk, personal statements, factual questions, and nonsense are routed separately; retrieval runs for informational queries
- **Conversational memory**: respects recent turns when answering, but never invents facts
- **Streaming replies** with `st.write_stream`
- **Responsive dark UI** (native Streamlit layout — sidebar, suggestion cards, chat input) with theme tracked at `.streamlit/config.toml`
- **Persistent chat history** across sessions (sidebar switcher)
- **Test suite** with pytest (routing, memory, retrieval, vectorstore validation)

## Project Structure

```text
.
├── app.py                    # Streamlit entry point
├── core/                     # Business logic
│   ├── chat_state.py         # Multi-conversation session state
│   ├── config.py             # Prompts, constants
│   ├── memory.py             # Name/profile extraction, recent context
│   ├── retrieval.py          # Hybrid lexical+vector retrieval, window expansion
│   └── router.py             # Query classification + streaming response
├── ui/                       # Rendering modules
│   ├── composer.py           # Chat input
│   ├── header.py             # Hero / chat header / suggestion cards
│   ├── messages.py           # Chat messages + source references
│   ├── sidebar.py            # Sidebar navigation
│   └── styles.py             # Small responsive CSS layer
├── tests/                    # pytest suite
├── src/
│   ├── ingest.py             # Rebuild the vectorstore
│   ├── rag.py
│   └── retrieve.py
├── vectorstore/              # Prebuilt FAISS index + docstore + manifest
├── vectorstore_store.py      # Vectorstore loading/validation
├── dataset_download.py       # Dataset fetch utility
├── .streamlit/config.toml    # Tracked dark theme
├── requirements.txt
└── requirements-dev.txt      # Dev/test dependencies
```

## Local Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your Groq API key in a local `.env` file (see `.env.example`):

```env
GROQ_API_KEY=your_groq_api_key
```

## Run Locally

```bash
streamlit run app.py
```

## Development

Install dev dependencies and run the test suite:

```bash
pip install -r requirements-dev.txt
pytest
```

## Rebuild the Vectorstore

If you update the dataset or embedding model, rebuild the FAISS store from the project root:

```bash
python src/ingest.py
```

This regenerates the vectorstore files in `vectorstore/`. The committed index contains **46,079 passages** built from ~21,270 source texts using `paraphrase-multilingual-MiniLM-L12-v2` (chunk size 250, overlap 64).

If retrieval fails after an embedding change, rebuild the vectorstore before redeploying.

## Deployment Notes

- Keep `app.py` at the repository root for Streamlit Cloud.
- The prebuilt `vectorstore/` directory is committed so Streamlit Cloud can serve the index without rebuilding (note: `index.faiss` is ~67 MB, above GitHub’s 50 MB recommendation but within limits).
- `.streamlit/config.toml` is tracked on purpose; the rest of `.streamlit/` (e.g. secrets) is ignored.
- Never commit `.env` or other secrets (`.gitignore` covers them).
- `tests/`, `pytest.ini`, and `requirements-dev.txt` are included so CI or reviewers can run the suite.

## Data Source

The app uses the Nepali text dataset from Hugging Face:

- [Sakonii/nepalitext-language-model-dataset](https://huggingface.co/datasets/Sakonii/nepalitext-language-model-dataset)
