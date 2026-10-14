# Student RAG System

A Gradio-based document question-answering prototype using retrieval-augmented generation (RAG).

## Features
- Extract text from PDF and TXT files
- Split extracted text into overlapping chunks
- Generate sentence embeddings with `all-MiniLM-L6-v2`
- Index and retrieve document chunks with FAISS
- Integrate Hugging Face inference for answer generation

## Technology
Python, Gradio, Sentence Transformers, FAISS, LangChain, PyPDF2, and Hugging Face Inference.

## Setup
```bash
python -m venv .venv
pip install -r requirements.txt
python app.py
```
Activate your virtual environment before installing dependencies. Configure any model-provider credentials through environment variables.

## Limitations
This is an educational prototype. Retrieval and generated responses may be inaccurate. Text extraction of scanned/image-only PDFs is not guaranteed.

## Maintainer
[Muhammad Aqib Ali](https://github.com/MuhammadAqibAli53)
