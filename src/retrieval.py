"""
Hybrid Retrieval Module: FAISS Dense + BM25 Sparse with RRF Fusion and Language Filtering.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from src.index_build import MultilingualIndexManager, LanguageIndex, tokenize_text

class RetrievedPassage(BaseModel):
    id: str
    doc_id: str
    text: str
    language: str
    rrf_score: float
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rank: int = 0
    is_selected: int = 0
    source_query_id: Optional[str] = None

class RetrievalResult(BaseModel):
    query_text: str
    detected_language: str
    passages: List[RetrievedPassage]
    top_score: float
    retrieval_latency_ms: float

class HybridRetriever:
    def __init__(self, index_manager: MultilingualIndexManager, rrf_k: int = 60):
        self.index_manager = index_manager
        self.rrf_k = rrf_k

    def search_single_language(
        self,
        query_text: str,
        lang_idx: LanguageIndex,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Dense FAISS + Sparse BM25 search for a single language partition."""
        # 1. Dense Search
        query_embedding = self.index_manager.embedding_model.encode(
            [query_text], normalize_embeddings=True
        )
        query_vec = np.array(query_embedding, dtype=np.float32)
        
        # FAISS search
        dense_distances, dense_indices = lang_idx.faiss_index.search(query_vec, min(top_k, len(lang_idx.passages)))
        
        dense_ranks = {}
        dense_scores = {}
        for rank, idx in enumerate(dense_indices[0]):
            if idx != -1 and idx < len(lang_idx.passages):
                p_id = lang_idx.passages[idx]["id"]
                dense_ranks[p_id] = rank + 1
                dense_scores[p_id] = float(dense_distances[0][rank])

        # 2. Sparse BM25 Search
        query_tokens = tokenize_text(query_text)
        bm25_scores = lang_idx.bm25_index.get_scores(query_tokens)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k]

        sparse_ranks = {}
        sparse_scores = {}
        for rank, idx in enumerate(bm25_top_indices):
            score = bm25_scores[idx]
            if score > 0:  # Only count positive keyword matches
                p_id = lang_idx.passages[idx]["id"]
                sparse_ranks[p_id] = rank + 1
                sparse_scores[p_id] = float(score)

        # 3. Reciprocal Rank Fusion (RRF)
        all_passage_ids = set(dense_ranks.keys()).union(set(sparse_ranks.keys()))
        rrf_results = []

        # Lookup dict for passages by ID
        passage_by_id = {p["id"]: p for p in lang_idx.passages}

        for p_id in all_passage_ids:
            score_rrf = 0.0
            if p_id in dense_ranks:
                score_rrf += 1.0 / (self.rrf_k + dense_ranks[p_id])
            if p_id in sparse_ranks:
                score_rrf += 1.0 / (self.rrf_k + sparse_ranks[p_id])

            passage_meta = passage_by_id[p_id]
            rrf_results.append({
                "passage_data": passage_meta,
                "rrf_score": score_rrf,
                "dense_score": dense_scores.get(p_id),
                "sparse_score": sparse_scores.get(p_id)
            })

        # Sort by RRF score descending
        rrf_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return rrf_results

    def retrieve(
        self,
        query_text: str,
        detected_language: str = "hi",
        top_n: int = 5
    ) -> RetrievalResult:
        """
        Executes pre-retrieval language-filtered hybrid retrieval with RRF fusion.
        Times the retrieval process to measure latency against the 200ms target.
        """
        t0 = time.time()
        
        # Ensure embedding model is loaded
        self.index_manager.load_embedding_model()

        # Pre-retrieval language filter: determine target language partitions
        lang_code = detected_language.lower()
        if lang_code.startswith("hi"):
            target_langs = ["hi"]
        elif lang_code.startswith("en"):
            target_langs = ["en"]
        else:
            # Fallback: search both available partitions if language unknown
            target_langs = [lang for lang in self.index_manager.indices.keys()]

        combined_candidates = []
        for lang in target_langs:
            if lang in self.index_manager.indices:
                lang_idx = self.index_manager.indices[lang]
                results = self.search_single_language(query_text, lang_idx, top_k=20)
                combined_candidates.extend(results)

        # Sort combined candidates by RRF score descending
        combined_candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
        final_candidates = combined_candidates[:top_n]

        retrieved_passages = []
        for rank, item in enumerate(final_candidates):
            pdata = item["passage_data"]
            retrieved_passages.append(RetrievedPassage(
                id=pdata["id"],
                doc_id=pdata.get("doc_id", pdata["id"]),
                text=pdata["text"],
                language=pdata.get("language", detected_language),
                rrf_score=item["rrf_score"],
                dense_score=item.get("dense_score"),
                sparse_score=item.get("sparse_score"),
                rank=rank + 1,
                is_selected=pdata.get("is_selected", 0),
                source_query_id=pdata.get("source_query_id")
            ))

        latency_ms = (time.time() - t0) * 1000.0
        top_score = retrieved_passages[0].rrf_score if retrieved_passages else 0.0

        return RetrievalResult(
            query_text=query_text,
            detected_language=detected_language,
            passages=retrieved_passages,
            top_score=top_score,
            retrieval_latency_ms=latency_ms
        )
