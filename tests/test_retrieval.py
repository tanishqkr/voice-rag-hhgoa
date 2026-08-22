"""
Unit Tests for Indexing & Retrieval Module (Offline, pytest).
Verifies FAISS + BM25 indexing, RRF Fusion math, and pre-retrieval language filtering.
"""

import pytest
from src.index_build import MultilingualIndexManager, LanguageIndex
from src.retrieval import HybridRetriever, RetrievalResult

@pytest.fixture(scope="module")
def mock_index_manager():
    manager = MultilingualIndexManager()
    
    hi_passages = [
        {"id": "hi_1", "doc_id": "d1", "text": "भारत की राजधानी नई दिल्ली है।", "language": "hi"},
        {"id": "hi_2", "doc_id": "d2", "text": "मुंबई महाराष्ट्र की आर्थिक राजधानी है।", "language": "hi"},
        {"id": "hi_3", "doc_id": "d3", "text": "चाय भारत का लोकप्रिय पेय है।", "language": "hi"}
    ]
    
    en_passages = [
        {"id": "en_1", "doc_id": "d4", "text": "New Delhi is the capital of India.", "language": "en"},
        {"id": "en_2", "doc_id": "d5", "text": "Mumbai is the financial capital of India.", "language": "en"},
        {"id": "en_3", "doc_id": "d6", "text": "Tea is a popular beverage in India.", "language": "en"}
    ]
    
    manager.build_index_for_language("hi", hi_passages)
    manager.build_index_for_language("en", en_passages)
    return manager

def test_language_filtered_retrieval(mock_index_manager):
    retriever = HybridRetriever(mock_index_manager, rrf_k=60)
    
    # Hindi Query -> Should retrieve ONLY Hindi passages
    res_hi = retriever.retrieve(query_text="भारत की राजधानी क्या है?", detected_language="hi", top_n=2)
    assert isinstance(res_hi, RetrievalResult)
    assert res_hi.detected_language == "hi"
    assert len(res_hi.passages) > 0
    assert all(p.language == "hi" for p in res_hi.passages)
    assert any("नई दिल्ली" in p.text or "राजधानी" in p.text for p in res_hi.passages)

    # English Query -> Should retrieve ONLY English passages
    res_en = retriever.retrieve(query_text="What is the capital of India?", detected_language="en", top_n=2)
    assert res_en.detected_language == "en"
    assert len(res_en.passages) > 0
    assert all(p.language == "en" for p in res_en.passages)
    assert any("New Delhi" in p.text or "capital" in p.text for p in res_en.passages)

def test_rrf_fusion_logic(mock_index_manager):
    retriever = HybridRetriever(mock_index_manager, rrf_k=60)
    res = retriever.retrieve(query_text="Tea beverage", detected_language="en", top_n=3)
    assert len(res.passages) > 0
    assert any("Tea" in p.text for p in res.passages)
    assert 0.0 < res.passages[0].rrf_score <= 2.0 / 61.0
