# Voice-Enabled Multilingual RAG System (Hindi & English)

A high-performance, voice-enabled Retrieval-Augmented Generation (RAG) system built for **HH Goa 2026 Shortlisting Task 2**.

---

## 1. Architecture Overview

```
Voice Input (User speech) 
  └─► Sarvam STT (transcribe + detect language: hi / en)
        └─► Input Guardrail (off-topic cosine distance & safety check)
              ├─► [Refusal: Query outside dataset scope]
              └─► Hybrid Retrieval (Language-filtered search space)
                    ├─► Dense Search: FAISS flat index (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
                    ├─► Sparse Search: BM25 (rank_bm25)
                    └─► Fusion: Reciprocal Rank Fusion (RRF)
                          └─► Retrieval Confidence Gate (Top RRF score threshold)
                                ├─► [Refusal: No strong match in dataset]
                                └─► Groq LLM Generation (Llama 3.1 8B Instant)
                                      └─► Output Groundedness Guardrail (Context verification)
                                            ├─► [Refusal: Insufficient grounded evidence]
                                            └─► Grounded Answer + Cited Passages
```

---

## 2. Technical Stack & Design Rationale

| Component | Technology | Rationale |
|---|---|---|
| **Speech-to-Text** | Sarvam AI (`saaras:v1`, mode=`transcribe`) | Purpose-built for Indic languages with native language detection and low latency. |
| **Embeddings** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Run locally on CPU (~10-30ms inference). BGE-M3 was deliberately avoided due to high CPU latency on cloud host limits. |
| **Dense Index** | FAISS (`faiss-cpu`) | In-memory flat/HNSW vector index with zero network overhead. |
| **Sparse Index** | BM25 (`rank_bm25`) | Catches exact keyword/entity matches that dense embeddings miss. |
| **Fusion** | Reciprocal Rank Fusion (RRF) | Combines dense and sparse rank signals seamlessly. |
| **Generation** | Groq API (`llama-3.1-8b-instant`) | Ultra-fast LPU inference enabling sub-second end-to-end response generation. |
| **Data Contracts & Retries** | Pydantic v2 & `tenacity` | Strictly typed stage boundaries and robust exponential backoff. |

---

## 3. Chunking & Retrieval Strategies

1. **Passage-level Chunks**: Natural unit for MSMARCO-XI (~50-150 words per passage).
2. **Semantic Chunking**: Embedding cosine-distance breakpoint splitting for long multi-passage documents.
3. **Metadata-Aware, Language-Filtered Indexing**: Chunks pre-tagged with language (`hi`/`en`). Sarvam STT language output filters search space *prior* to dense/sparse retrieval, improving both retrieval speed and cross-lingual accuracy.

---

## 4. Guardrails Specification

1. **Input Guardrail**: Off-topic cosine similarity gate + regex safety blocklist.
2. **Retrieval Confidence Gate**: Refuses weak context if top RRF score falls below empirical threshold.
3. **Output Groundedness Guardrail**: Validates that generated answers rely strictly on retrieved context passages.

---

## 5. Latency Target Interpretation

The 200ms target in the task brief explicitly applies to the **retrieval stage** (Query Embedding → FAISS & BM25 Search → RRF Fusion). STT (Sarvam) and Generation (Groq) are network-bound external API calls and are measured and reported separately in full benchmark metrics.

---

## 6. Quick Start

### Prerequisites
- Python 3.11+
- Sarvam API Key
- Groq API Key

### Setup
```bash
# Clone and set up environment
git clone <repo-url>
cd voice-rag-hhgoa

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and insert your SARVAM_API_KEY and GROQ_API_KEY

# Download and sample dataset (10,000 passages per language)
python3 src/download_data.py
```

### Running the App
```bash
streamlit run app.py
```
