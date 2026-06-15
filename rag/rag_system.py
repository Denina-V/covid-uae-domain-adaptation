"""
RAG System: COVID Research Q&A
-------------------------------
Retrieves relevant research chunks via sentence-transformer embeddings (FAISS),
then passes context + question to Claude for a grounded answer.
"""

import os
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from knowledge_base import DOCUMENTS

# ── Embedding (local, no API needed for retrieval) ─────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    import faiss

    _model = SentenceTransformer("all-MiniLM-L6-v2")
    _texts = [d["text"] for d in DOCUMENTS]
    _embeddings = _model.encode(_texts, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(_embeddings)
    _index = faiss.IndexFlatIP(_embeddings.shape[1])
    _index.add(_embeddings)
    _USE_EMBEDDINGS = True
except Exception:
    _USE_EMBEDDINGS = False


def retrieve(query: str, k: int = 3) -> list[dict]:
    """Return top-k most relevant documents for the query."""
    if _USE_EMBEDDINGS:
        q_vec = _model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_vec)
        _, indices = _index.search(q_vec, k)
        return [DOCUMENTS[i] for i in indices[0]]
    # Fallback: keyword overlap
    scores = []
    q_words = set(query.lower().split())
    for doc in DOCUMENTS:
        overlap = len(q_words & set(doc["text"].lower().split()))
        scores.append((overlap, doc))
    scores.sort(key=lambda x: -x[0])
    return [d for _, d in scores[:k]]


def answer(question: str, model_results: dict | None = None) -> str:
    """
    Retrieve relevant docs and ask Claude to answer grounded in them.
    model_results: optional dict with baseline/adapted MAE, R2 to include as context.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Add it to your .env file:\n"
            "  ANTHROPIC_API_KEY=sk-ant-..."
        )

    import anthropic

    docs = retrieve(question, k=3)
    context = "\n\n".join(
        f"[{d['title']}]\n{d['text']}" for d in docs
    )

    model_ctx = ""
    if model_results:
        model_ctx = (
            f"\n\nModel results from this project:\n"
            f"- Baseline (source-only) MAE: {model_results['baseline']['mae']}, "
            f"R²: {model_results['baseline']['r2']}\n"
            f"- Adapted (+ UAE labels) MAE: {model_results['adapted']['mae']}, "
            f"R²: {model_results['adapted']['r2']}\n"
            f"- MAE improvement after adaptation: {model_results['improvement_pct']}%"
        )

    prompt = f"""You are a research assistant explaining COVID-19 modelling findings to a researcher.
Answer the question using ONLY the provided research context. Be concise and precise.

Research Context:
{context}
{model_ctx}

Question: {question}

Answer:"""

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


if __name__ == "__main__":
    questions = [
        "Why is UAE's reproduction rate harder to predict than Germany's?",
        "What is semi-supervised learning and why does it help here?",
        "How much labeled data do we need for domain adaptation to work?",
    ]

    # Load model results if available
    try:
        with open("../models/results.json") as f:
            model_results = json.load(f)
    except FileNotFoundError:
        model_results = None

    print("=" * 60)
    for q in questions:
        print(f"\nQ: {q}")
        print("-" * 60)
        try:
            a = answer(q, model_results=model_results)
            print(f"A: {a}")
        except EnvironmentError as e:
            print(f"[RAG skipped — {e}]")
        print()
