# COVID-19 UAE: Domain Adaptation + RAG Pipeline

**UAE reproduction rate prediction** demonstrating why domain adaptation and semi-supervised learning matter for under-resourced health systems.

---

## Project Structure

```
├── covid_uae_eda.ipynb                        # EDA: missingness & domain shift (start here)
├── notebooks/
│   └── domain_adaptation_pipeline.ipynb       # ML pipeline + RAG system
├── models/
│   └── train_model.py                         # Baseline vs adapted model training
├── rag/
│   ├── knowledge_base.py                      # Curated COVID research summaries
│   └── rag_system.py                          # Retrieval + Claude-powered Q&A
├── data/
│   ├── covid_uae_comparison.csv               # UAE, Germany, India, Singapore
│   └── generate_data.py                       # Regenerate dataset
├── .env.example                               # API key template
└── requirements.txt
```

---

## Three-Component Pipeline

### 1. EDA (`covid_uae_eda.ipynb`)
- Raw vs smoothed signal
- UAE missingness heatmap — ~39% of R-values missing in **clusters**
- Cross-country comparison — visual domain shift

### 2. Domain Adaptation (`notebooks/domain_adaptation_pipeline.ipynb`)
| Model | MAE | R² |
|---|---|---|
| Baseline (source-only: Germany/India/Singapore) | ~0.49 | ~-0.03 |
| Adapted (+10% UAE labeled data) | ~0.23 | ~0.76 |

**54% MAE improvement** using just 10% UAE labels — this is the domain adaptation result.

### 3. RAG System (`rag/rag_system.py`)
Retrieves relevant COVID research via sentence-transformer embeddings (FAISS), then uses **Claude** to answer questions grounded in that context. Example queries:
- *"Why is UAE's reproduction rate harder to predict?"*
- *"How much labeled data do we need for domain adaptation to work?"*
- *"What is semi-supervised learning and why does it help here?"*

---

## Quickstart

```bash
pip install -r requirements.txt

# (Optional) Add Anthropic API key for RAG
cp .env.example .env
# Edit .env → add ANTHROPIC_API_KEY=sk-ant-...

# Run EDA
jupyter notebook covid_uae_eda.ipynb

# Run full pipeline
jupyter notebook notebooks/domain_adaptation_pipeline.ipynb
```

---

## Key Findings

1. UAE's R-value is **39% missing in clusters** — not random noise, a structural labelling problem
2. A model trained on Germany/India/Singapore **fails on UAE** (R² ≈ -0.03)
3. Adding just **10% UAE labeled observations** recovers 54% of prediction error
4. This validates the SSL + domain adaptation direction for data-scarce health systems
5. A RAG system grounds model explanations in peer-reviewed research
