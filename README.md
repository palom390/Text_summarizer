# Text Summarizer

Simple Streamlit app that uses Hugging Face `transformers` (BART) to summarize long text.

Quick start (local):

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Docker (build & run):

```bash
docker build -t text-summarizer:latest .
docker run -p 8501:8501 --rm text-summarizer:latest
```

Notes:
- The container downloads model weights on first run; ensure sufficient disk and memory.
- For production-scale usage, consider hosting the model using Hugging Face Inference API or using a GPU-enabled instance.

Files:
- `app.py` — Streamlit UI and model-loading wrapper
- `summarizer.py` — core summarization helpers (optional)
- `requirements.txt` — Python dependencies
