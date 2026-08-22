"""
Unit Tests for Pipeline Harness (Offline, pytest with fixtures).
"""

import pytest
from src.index_build import MultilingualIndexManager
from src.retrieval import HybridRetriever
from src.harness import PipelineHarness, PipelineOutput

@pytest.fixture(scope="module")
def mock_retriever():
    manager = MultilingualIndexManager()
    hi_passages = [{"id": "hi_1", "doc_id": "d1", "text": "भारत की राजधानी नई दिल्ली है।", "language": "hi"}]
    manager.build_index_for_language("hi", hi_passages)
    return HybridRetriever(manager)

def test_harness_fixture_pipeline(mock_retriever):
    harness = PipelineHarness(retriever=mock_retriever, use_fixtures=True)
    out = harness.execute_pipeline(text_query="भारत की राजधानी क्या है?", forced_language="hi")
    
    assert isinstance(out, PipelineOutput)
    assert out.query_text == "भारत की राजधानी क्या है?"
    assert out.detected_language == "hi"
    assert out.retrieval is not None
    assert len(out.retrieval.passages) > 0
    assert out.generation is not None
    assert out.generation.is_mock is True
    assert "retrieval_ms" in out.stage_latencies
    assert out.refusal is False
