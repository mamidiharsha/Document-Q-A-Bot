# RAG Document Q&A Bot

A retrieval-augmented generation (RAG) document Q&A bot built with Python.
It indexes PDF, TXT, and DOCX documents into ChromaDB, then answers questions
with Google Gemini embeddings and generation.


## Setup

1. Create a Python virtual environment (recommended):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # PowerShell
```

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Copy the environment template and set your API key:

```bash
copy .env.example .env
```

Update `.env` with a valid `GOOGLE_API_KEY`.

## Documents

The repository includes source text files in `data/` and the generated documents:

- `data/coffee_history.pdf`
- `data/renewable_energy.pdf`
- `data/solar_system.docx`

Supported input formats:
- PDF
- TXT
- DOCX

## Usage

### Generate missing document formats

If you ever need to regenerate `PDF` / `DOCX` files:

```bash
python convert_docs.py
```

### Index documents

```bash
python index.py
```

To force re-indexing:

```bash
python index.py --force
```

### Use the CLI

```bash
python cli.py
```

### Run the Streamlit web UI

```bash
streamlit run app.py
```

## Project Components

- `src/config.py` — environment and project configuration
- `src/document_loader.py` — loads PDF, TXT, DOCX contents
- `src/chunker.py` — recursive character chunking
- `src/embedder.py` — Gemini embedding service wrapper
- `src/vector_store.py` — ChromaDB persistent storage wrapper
- `src/retriever.py` — top-K retrieval pipeline
- `src/generator.py` — Gemini answer generation with citation instructions
- `src/pipeline.py` — end-to-end orchestration

## Notes

- `chroma_db/` stores persistent vector data in `chroma.sqlite3`.


