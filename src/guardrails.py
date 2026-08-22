"""
Guardrails Module for Input Off-Topic/Safety Check, Retrieval Confidence Gate,
and Output Groundedness Verification.
"""

import re
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from src.retrieval import RetrievedPassage

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CONFIG_PATH = DATA_DIR / "processed" / "guardrail_config.json"

UNSAFE_KEYWORDS = [
    "hack", "exploit", "bomb", "kill", "suicide", "malware", "virus",
    "हत्या", "बम", "आत्महत्या", "हैक"
]

class GuardrailVerdict(BaseModel):
    passed: bool
    check_type: str  # "input_safety", "retrieval_confidence", "output_groundedness"
    score: float
    threshold: float
    reason: str

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
        
        # Load empirical thresholds if present
        self.load_config()

    def load_config(self, config_path: Path = CONFIG_PATH):
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    self.input_cos_threshold = cfg.get("input_cos_threshold", self.input_cos_threshold)
                    self.retrieval_conf_threshold = cfg.get("retrieval_conf_threshold", self.retrieval_conf_threshold)
                    self.groundedness_sim_threshold = cfg.get("groundedness_sim_threshold", self.groundedness_sim_threshold)
            except Exception as e:
                pass

    def check_input_guardrail(self, query_text: str, language: str = "hi") -> GuardrailVerdict:
        """Input Guardrail: Keyword safety check & Domain semantic relevance."""
        # 1. Keyword Blocklist Check
        query_lower = query_text.lower()
        for kw in UNSAFE_KEYWORDS:
            if kw in query_lower:
                return GuardrailVerdict(
                    passed=False,
                    check_type="input_safety",
                    score=1.0,
                    threshold=0.0,
                    reason=f"Unsafe keyword detected: '{kw}'"
                )

        # 2. Embedding Cosine Similarity Domain Check
        if self.embedding_model:
            sample_in_domain = "कॉर्पोरेशन या कंपनी क्या है?" if language == "hi" else "What is a corporation or business entity?"
            embeddings = self.embedding_model.encode([query_text, sample_in_domain], normalize_embeddings=True)
            cos_sim = float(np.dot(embeddings[0], embeddings[1]))
            
            if cos_sim < self.input_cos_threshold:
                return GuardrailVerdict(
                    passed=False,
                    check_type="input_safety",
                    score=cos_sim,
                    threshold=self.input_cos_threshold,
                    reason=f"Off-topic query (semantic similarity {cos_sim:.3f} < threshold {self.input_cos_threshold:.3f})"
                )
            return GuardrailVerdict(
                passed=True,
                check_type="input_safety",
                score=cos_sim,
                threshold=self.input_cos_threshold,
                reason="Query passed safety & domain relevance checks."
            )

        return GuardrailVerdict(
            passed=True,
            check_type="input_safety",
            score=1.0,
            threshold=self.input_cos_threshold,
            reason="Input safety passed (no embedding model bound)."
        )

    def check_retrieval_confidence(self, top_rrf_score: float) -> GuardrailVerdict:
        """Retrieval Confidence Gate: Verifies top retrieved RRF score exceeds threshold."""
        passed = top_rrf_score >= self.retrieval_conf_threshold
        reason = "Retrieval confidence sufficient." if passed else f"Low retrieval confidence (top RRF score {top_rrf_score:.4f} < threshold {self.retrieval_conf_threshold:.4f})"
        return GuardrailVerdict(
            passed=passed,
            check_type="retrieval_confidence",
            score=top_rrf_score,
            threshold=self.retrieval_conf_threshold,
            reason=reason
        )

    def check_output_groundedness(
        self,
        query_text: str,
        generated_answer: str,
        retrieved_passages: List[RetrievedPassage]
    ) -> GuardrailVerdict:
        """Output Groundedness Guardrail: Checks semantic alignment between answer and context."""
        refusal_phrases = ["cannot answer", "not enough information", "dataset", "उपलब्ध डेटासेट", "उत्तर नहीं"]
        ans_lower = generated_answer.lower()
        if any(phrase in ans_lower for phrase in refusal_phrases):
            return GuardrailVerdict(
                passed=False,
                check_type="output_groundedness",
                score=0.0,
                threshold=self.groundedness_sim_threshold,
                reason="Model issued grounded refusal phrase."
            )

        if self.embedding_model and retrieved_passages:
            context_text = " ".join([p.text for p in retrieved_passages[:3]])
            embs = self.embedding_model.encode([generated_answer, context_text], normalize_embeddings=True)
            overlap_score = float(np.dot(embs[0], embs[1]))

            passed = overlap_score >= self.groundedness_sim_threshold
            reason = "Answer grounded in context." if passed else f"Answer ungrounded (semantic overlap {overlap_score:.3f} < threshold {self.groundedness_sim_threshold:.3f})"
            return GuardrailVerdict(
                passed=passed,
                check_type="output_groundedness",
                score=overlap_score,
                threshold=self.groundedness_sim_threshold,
                reason=reason
            )

        return GuardrailVerdict(
            passed=True,
            check_type="output_groundedness",
            score=1.0,
            threshold=self.groundedness_sim_threshold,
            reason="Output groundedness passed."
        )
