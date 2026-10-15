# Student RAG System
### Document intelligence · Semantic search · Generative AI

A document-grounded question-answering prototype built to help students retrieve relevant information from their own study materials rather than relying solely on a language model's general knowledge.

**Skills demonstrated:** Python · Retrieval-Augmented Generation · NLP · Vector Search · Document Processing · Model Integration · Gradio

## Engineering highlights
| Challenge | Implementation | Skill showcased |
|---|---|---|
| Extract usable study content | PDF and TXT readers using PyPDF2 and Python I/O | Document parsing and error handling |
| Handle long documents | `RecursiveCharacterTextSplitter` (1,000-character chunks, 200-character overlap) | Context-window-aware processing |
| Search by meaning | `all-MiniLM-L6-v2` Sentence Transformer embeddings | Semantic representation |
| Efficient similarity search | FAISS `IndexFlatIP` with normalized vectors | Vector indexing / cosine similarity |
| Connect retrieval to generation | Hugging Face `InferenceClient` integration | LLM service integration |
| Make the pipeline accessible | Gradio interface | AI application development |

## Architecture
```text
User uploads PDF/TXT
        |
    Text extraction
        |
  Chunking + metadata
        |
 Sentence embeddings
        |
   FAISS vector index
        |
Question -> Query embedding -> Top matching chunks
                              |
                        Model inference
                              |
                          Answer / UI
```

## What this project demonstrates
- Separating ingestion, indexing, retrieval, and response-generation responsibilities.
- Retaining source file and chunk metadata for retrieved passages.
- Normalizing embedding vectors before inner-product similarity search.
- Building a usable interface around machine-learning components.

## Technology
Python, Gradio, NumPy, Pandas, Sentence Transformers, FAISS, PyPDF2, LangChain, Hugging Face Hub.

## Running locally
```bash
python -m venv .venv
# Activate the environment for your operating system
pip install -r requirements.txt
python app.py
```
Configure the model provider's credentials where required. Do not commit API keys.

## Status and limitations
Educational prototype; generated answers can be wrong. Scanned PDFs may require separate OCR preprocessing. No independently verified accuracy benchmark is claimed.

**Maintained by [Muhammad Aqib Ali](https://github.com/MuhammadAqibAli53).**
