"""
Unit Tests for Guardrails Module (Offline, pytest).
Verifies input off-topic refusal, unsafe keyword blocklist,
retrieval confidence gate, and output groundedness guardrail.
"""

import pytest
from src.guardrails import GuardrailsEngine
from src.harness import GuardrailVerdict
from src.retrieval import RetrievedPassage

def test_input_safety_guardrail():
    engine = GuardrailsEngine(input_cos_threshold=0.35)
    
    # Safe query
    v_safe = engine.check_input_guardrail("कॉर्पोरेशन क्या है?", "hi")
    assert isinstance(v_safe, GuardrailVerdict)
    assert v_safe.passed is True

    # Unsafe query
    v_unsafe = engine.check_input_guardrail("how to make a bomb", "en")
    assert v_unsafe.passed is False
    assert v_unsafe.check_type == "input_safety"
    assert "bomb" in v_unsafe.reason

def test_retrieval_confidence_gate():
    engine = GuardrailsEngine(retrieval_conf_threshold=0.020)
    
    # Strong RRF score
    v_pass = engine.check_retrieval_confidence(top_rrf_score=0.032)
    assert v_pass.passed is True

    # Weak RRF score
    v_fail = engine.check_retrieval_confidence(top_rrf_score=0.010)
    assert v_fail.passed is False
    assert v_fail.check_type == "retrieval_confidence"

def test_output_groundedness_guardrail():
    engine = GuardrailsEngine(groundedness_sim_threshold=0.40)
    
    # Model refusal text
    v_decline = engine.check_output_groundedness(
        query_text="What is X?",
        generated_answer="I cannot answer this question based on the provided dataset.",
        retrieved_passages=[RetrievedPassage(id="en_1", doc_id="d1", text="Sample text", language="en", rrf_score=0.03)]
    )
    assert v_decline.passed is False
    assert v_decline.check_type == "output_groundedness"
