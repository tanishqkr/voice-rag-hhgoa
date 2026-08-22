# Voice-Enabled Multilingual RAG System (Hindi & English)

A high-performance, voice-enabled Retrieval-Augmented Generation (RAG) system built for **HH Goa 2026 Shortlisting Task 2**.

---

## 1. Architecture Overview

```
Voice Input (User speech) 
  └─► Sarvam STT saarika:v2.5 (transcribe + detect language: hi / en)
        └─► Input Guardrail (off-topic corpus cosine distance & safety check)
              ├─► [Refusal: Query outside dataset scope]
              └─► Hybrid Retrieval (Language-filtered search space)
                    ├─► Dense Search: FAISS flat index (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
                    ├─► Sparse Search: BM25 (rank_bm25)
                    └─► Fusion: Reciprocal Rank Fusion (RRF)
                          └─► Retrieval Confidence Gate (Top RRF score threshold)
                                ├─► [Refusal: No strong match in dataset]
                                └─► Groq LLM Generation (openai/gpt-oss-20b)
                                      └─► Output Groundedness Guardrail (Context verification)
                                            ├─► [Refusal: Insufficient grounded evidence]
                                            └─► Grounded Answer + Cited Passages
```

---

## 2. Technical Stack & Design Rationale

| Component | Technology | Rationale |
|---|---|---|
| **Speech-to-Text** | Sarvam AI (`saarika:v2.5`) | Purpose-built for Indic languages with native language detection and low latency. Dynamic magic-bytes MIME auto-detection for browser WebM/Ogg mic recording. |
| **Embeddings** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Run locally on CPU (~10-30ms inference). BGE-M3 was deliberately avoided due to high CPU latency on cloud host limits. |
| **Dense Index** | FAISS (`faiss-cpu`) | In-memory flat inner-product vector index with zero network overhead. |
| **Sparse Index** | BM25 (`rank_bm25`) | Catches exact keyword/entity matches that dense embeddings miss. |
| **Fusion** | Reciprocal Rank Fusion (RRF) | Combines dense and sparse rank signals seamlessly ($k=60$). |
| **Generation** | Groq API (`openai/gpt-oss-20b`) | Non-agentic text LLM with high RPM ceiling (1,000 RPM) enabling fast response generation. |
| **Context Tradeoff** | Top-3 Prompt Context Window | To cut prompt token consumption and generation latency, the top 3 retrieved passages are passed into the LLM prompt context, while all top 5 retrieved candidates are displayed in the UI. |
| **Data Contracts & Retries** | Pydantic v2 & `tenacity` | Strictly typed stage boundaries and robust exponential backoff. |

---

## 3. Empirical Latency Benchmark Summary (50 Held-Out Queries)

Results from `data/processed/latency_results.jsonl` processed across 50 held-out queries (30 in-domain, 20 off-topic):

| Pipeline Stage | P50 (ms) | P70 (ms) | P100 (ms) | Note / Target Status |
|---|---|---|---|---|
| **Retrieval-Only (FAISS+BM25)** | **77.9** | **84.9** | **184.4** | Target < 200 ms (**PASS**) |
| **STT (Sarvam)** | 0.1 | 0.1 | 2.6 | External API Call |
| **Generation (Groq `openai/gpt-oss-20b`)** | **665.0** | **758.6** | **2313.2** | External API Call (0 Rate-Limit Retries) |
| **Full End-to-End** | **726.3** | **823.9** | **2393.3** | Pipeline Total |

- **Total Queries Processed**: 50
- **Rate Limit Retries / 429 Errors**: 0
- **P100 / Median Ratio**: $3.47\times$ (No 10x+ throttling outliers)

---

## 4. Chunking & Retrieval Strategies

1. **Passage-level Chunks**: Natural unit for MSMARCO-XI (~50-150 words per passage).
2. **Semantic Chunking**: Embedding cosine-distance breakpoint splitting for long multi-passage documents.
3. **Metadata-Aware, Language-Filtered Indexing**: Chunks pre-tagged with language (`hi`/`en`). Sarvam STT language output filters search space *prior* to dense/sparse retrieval, improving both retrieval speed and cross-lingual accuracy.

---

## 5. Guardrails Specification

1. **Input Guardrail**: Off-topic corpus cosine similarity gate (`threshold = 0.520`, **91.23% F1 score**) + regex safety blocklist.
2. **Retrieval Confidence Gate**: Refuses weak context if top RRF score falls below empirical threshold (`0.015`).
3. **Output Groundedness Guardrail**: Validates that generated answers rely strictly on retrieved context passages (`threshold = 0.400`).

---

## 6. Latency Target Interpretation

The 200ms target in the task brief explicitly applies to the **retrieval stage** (Query Embedding → FAISS & BM25 Search → RRF Fusion). STT (Sarvam) and Generation (Groq) are network-bound external API calls and are measured and reported separately in full benchmark metrics.

---

## 7. Quick Start

### Prerequisites
- Python 3.11+
- Sarvam API Key
- Groq API Key

### Setup
```bash
# Clone and set up environment
git clone https://github.com/tanishqkr/voice-rag-hhgoa.git
cd voice-rag-hhgoa

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and insert your SARVAM_API_KEY and GROQ_API_KEY

# Download and sample dataset (10,000 passages per language)
python3 src/download_data.py
```

### Running the Benchmark
```bash
PYTHONPATH=. python3 src/latency_bench.py
```

### Running the App
```bash
streamlit run app.py
```
