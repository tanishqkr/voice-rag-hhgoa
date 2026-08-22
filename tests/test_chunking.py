"""
Unit Tests for Chunking Processor (Offline, pytest).
"""

import pytest
from src.chunking import ChunkingProcessor, Chunk

def test_passage_chunking():
    processor = ChunkingProcessor()
    passage_dict = {
        "id": "hi_p_1",
        "doc_id": "doc_100",
        "text": "भारत एक विविध संस्कृतियों का देश है।",
        "language": "hi",
        "is_selected": 1,
        "source_query_id": "q_1"
    }
    chunk = processor.passage_chunking(passage_dict)
    assert isinstance(chunk, Chunk)
    assert chunk.chunk_id == "hi_p_1"
    assert chunk.language == "hi"
    assert chunk.chunk_type == "passage"
    assert chunk.source_query_id == "q_1"
    assert chunk.metadata["is_selected"] == 1

def test_semantic_chunking_fallback():
    processor = ChunkingProcessor(embedding_model=None)
    doc_text = (
        "This is sentence one. This is sentence two. This is sentence three. "
        "This is sentence four. This is sentence five."
    )
    chunks = processor.semantic_chunking(doc_text, doc_id="doc_test", language="en")
    assert len(chunks) == 2
    assert chunks[0].chunk_type == "semantic"
    assert chunks[0].language == "en"
    assert "sentence one" in chunks[0].text

def test_process_passages_batch():
    processor = ChunkingProcessor()
    passages = [
        {"id": "en_1", "text": "Passage 1", "language": "en"},
        {"id": "hi_1", "text": "मार्ग 1", "language": "hi"}
    ]
    chunks = processor.process_passages_batch(passages)
    assert len(chunks) == 2
    assert chunks[0].language == "en"
    assert chunks[1].language == "hi"
