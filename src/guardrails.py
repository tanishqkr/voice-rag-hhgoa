"""
Guardrails Module for Input Off-Topic/Safety Check, Retrieval Confidence Gate,
and Output Groundedness Verification.
"""

import re
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.harness import GuardrailVerdict
from src.retrieval import RetrievedPassage

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CONFIG_PATH = DATA_DIR / "processed" / "guardrail_config.json"

UNSAFE_KEYWORDS = [
    "hack", "exploit", "bomb", "kill", "suicide", "malware", "virus",
    "हत्या", "बम", "आत्महत्या", "हैक"
]

class GuardrailsEngine:
    def __init__(
        self,
        embedding_model=None,
        input_cos_threshold: float = 0.35,
        retrieval_conf_threshold: float = 0.025,
        groundedness_sim_threshold: float = 0.40
    ):
        self.embedding_model = embedding_model
        self.input_cos_threshold = input_cos_threshold
        self.retrieval_conf_threshold = retrieval_conf_threshold
        self.groundedness_sim_threshold = groundedness_sim_threshold
        
        # Load calibrated parameters if file exists
        self.load_calibrated_config()

    def load_calibrated_config(self, config_path: Path = CONFIG_PATH):
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                self.input_cos_threshold = cfg.get("input_cos_threshold", self.input_cos_threshold)
                self.retrieval_conf_threshold = cfg.get("retrieval_conf_threshold", self.retrieval_conf_threshold)
                self.groundedness_sim_threshold = cfg.get("groundedness_sim_threshold", self.groundedness_sim_threshold)
                print(f"✅ Loaded calibrated guardrail thresholds: input={self.input_cos_threshold:.3f}, conf={self.retrieval_conf_threshold:.4f}, ground={self.groundedness_sim_threshold:.3f}")
            except Exception as e:
                print(f"⚠️ Could not load guardrail_config.json: {e}")

    def check_input_guardrail(self, query_text: str, language: str = "hi", corpus_embeddings: Optional[np.ndarray] = None) -> GuardrailVerdict:
        """
        Checkpoint 1: Input Guardrail.
        1. Unsafe/inappropriate regex/keyword blocklist.
        2. Off-topic check via max cosine similarity against corpus.
        """
        # Safety Check
        query_lower = query_text.lower()
        for kw in UNSAFE_KEYWORDS:
            if kw in query_lower:
                return GuardrailVerdict(
                    passed=False,
                    check_type="input_safety",
                    reason=f"Safety blocklist triggered by keyword '{kw}'",
                    score=0.0
                )

        # Off-topic Cosine Distance Check
        if self.embedding_model is not None and corpus_embeddings is not None and len(corpus_embeddings) > 0:
            q_emb = self.embedding_model.encode([query_text], normalize_embeddings=True)[0]
            sims = np.dot(corpus_embeddings, q_emb)
            max_sim = float(np.max(sims))
            
            if max_sim < self.input_cos_threshold:
                return GuardrailVerdict(
                    passed=False,
                    check_type="input_offtopic",
                    reason=f"Query is off-topic (max cosine similarity {max_sim:.3f} < threshold {self.input_cos_threshold:.3f})",
                    score=max_sim
                )
            return GuardrailVerdict(
                passed=True,
                check_type="input_offtopic",
                reason="Query passed off-topic & safety checks",
                score=max_sim
            )

        # Fallback if no embeddings available
        return GuardrailVerdict(passed=True, check_type="input_offtopic", reason="Input passed basic check", score=1.0)

    def check_retrieval_confidence(self, top_rrf_score: float) -> GuardrailVerdict:
        """
        Checkpoint 2: Retrieval Confidence Gate.
        Refuses queries if top RRF score is below empirical threshold.
        """
        if top_rrf_score < self.retrieval_conf_threshold:
            return GuardrailVerdict(
                passed=False,
                check_type="retrieval_confidence",
                reason=f"Low retrieval confidence (top RRF score {top_rrf_score:.4f} < threshold {self.retrieval_conf_threshold:.4f})",
                score=top_rrf_score
            )
        return GuardrailVerdict(
            passed=True,
            check_type="retrieval_confidence",
            reason="Retrieval score meets confidence threshold",
            score=top_rrf_score
        )

    def check_output_groundedness(
        self,
        query_text: str,
        generated_answer: str,
        retrieved_passages: List[RetrievedPassage]
    ) -> GuardrailVerdict:
        """
        Checkpoint 3: Output Groundedness Guardrail.
        Verifies that generated answer has sufficient semantic overlap with retrieved context.
        """
        if not retrieved_passages:
            return GuardrailVerdict(passed=False, check_type="output_groundedness", reason="No retrieved passages to verify grounding", score=0.0)

        if "cannot answer this question based on the provided dataset" in generated_answer.lower():
            return GuardrailVerdict(passed=False, check_type="output_groundedness", reason="Model explicitly declined due to missing context", score=0.0)

        if self.embedding_model is not None:
            ans_emb = self.embedding_model.encode([generated_answer], normalize_embeddings=True)[0]
            context_texts = [p.text for p in retrieved_passages]
            ctx_embs = self.embedding_model.encode(context_texts, normalize_embeddings=True)
            
            sims = np.dot(ctx_embs, ans_emb)
            max_sim = float(np.max(sims))

            if max_sim < self.groundedness_sim_threshold:
                return GuardrailVerdict(
                    passed=False,
                    check_type="output_groundedness",
                    reason=f"Answer ungrounded (semantic overlap {max_sim:.3f} < threshold {self.groundedness_sim_threshold:.3f})",
                    score=max_sim
                )
            return GuardrailVerdict(
                passed=True,
                check_type="output_groundedness",
                reason="Answer is grounded in retrieved context",
                score=max_sim
            )

        return GuardrailVerdict(passed=True, check_type="output_groundedness", reason="Output passed basic check", score=1.0)
