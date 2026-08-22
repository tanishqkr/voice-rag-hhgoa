"""
Chunking Module for Voice-Enabled RAG.
Implements 3 chunking & metadata strategies:
1. Passage-level chunking (MSMARCO baseline unit).
2. Semantic chunking (embedding cosine-distance breakpoint splitting for long documents).
3. Metadata-aware tagging (language, document ID, and source query ID).
"""

from typing import List, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel, Field

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    language: str
    chunk_type: str = "passage"  # "passage" or "semantic"
    source_query_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ChunkingProcessor:
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model

    def passage_chunking(self, passage_data: Dict[str, Any]) -> Chunk:
        """
        Strategy 1: Passage-level chunking.
        Baseline chunking unit for short MSMARCO-XI passages.
        """
        chunk_id = passage_data.get("id") or f"p_{hash(passage_data.get('text', ''))}"
        doc_id = passage_data.get("doc_id", chunk_id)
        text = passage_data.get("text", "").strip()
        language = passage_data.get("language", "en")
        source_query_id = passage_data.get("source_query_id")
        
        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text,
            language=language,
            chunk_type="passage",
            source_query_id=source_query_id,
            metadata={
                "is_selected": passage_data.get("is_selected", 0),
                "word_count": len(text.split())
            }
        )

    def semantic_chunking(
        self,
        document_text: str,
        doc_id: str,
        language: str,
        breakpoint_percentile_threshold: float = 80.0,
        source_query_id: Optional[str] = None
    ) -> List[Chunk]:
        """
        Strategy 2: Semantic Chunking.
        Splits long concatenated document text into semantically cohesive chunks 
        by embedding sentences and detecting cosine-distance breakpoints.
        """
        if not document_text or not document_text.strip():
            return []
            
        # Split document into sentences (handles Hindi full stop '|' and English '.')
        import re
        sentences = [s.strip() for s in re.split(r'(?<=[.!?।])\s+', document_text) if s.strip()]
        
        if len(sentences) <= 1:
            return [Chunk(
                chunk_id=f"{doc_id}_sem_0",
                doc_id=doc_id,
                text=document_text.strip(),
                language=language,
                chunk_type="semantic",
                source_query_id=source_query_id,
                metadata={"sentence_count": len(sentences)}
            )]

        # If no embedding model provided, fall back to fixed sentence windowing
        if self.embedding_model is None:
            chunks = []
            group_size = 3
            for i in range(0, len(sentences), group_size):
                chunk_sentences = sentences[i:i + group_size]
                chunk_text = " ".join(chunk_sentences)
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_sem_{len(chunks)}",
                    doc_id=doc_id,
                    text=chunk_text,
                    language=language,
                    chunk_type="semantic",
                    source_query_id=source_query_id,
                    metadata={"sentence_count": len(chunk_sentences)}
                ))
            return chunks

        # Embed sentences using local sentence-transformers model
        embeddings = self.embedding_model.encode(sentences, normalize_embeddings=True)
        
        # Calculate cosine distances between consecutive sentences
        distances = []
        for i in range(len(embeddings) - 1):
            cosine_sim = np.dot(embeddings[i], embeddings[i + 1])
            cosine_dist = 1.0 - cosine_sim
            distances.append(cosine_dist)

        # Determine distance threshold based on percentile
        threshold = np.percentile(distances, breakpoint_percentile_threshold) if distances else 0.5
        
        # Group sentences into chunks based on breakpoint threshold
        chunks = []
        current_sentences = [sentences[0]]
        
        for i, dist in enumerate(distances):
            if dist > threshold:
                # Breakpoint reached
                chunk_text = " ".join(current_sentences)
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_sem_{len(chunks)}",
                    doc_id=doc_id,
                    text=chunk_text,
                    language=language,
                    chunk_type="semantic",
                    source_query_id=source_query_id,
                    metadata={"sentence_count": len(current_sentences), "breakpoint_dist": float(dist)}
                ))
                current_sentences = [sentences[i + 1]]
            else:
                current_sentences.append(sentences[i + 1])
                
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_sem_{len(chunks)}",
                doc_id=doc_id,
                text=chunk_text,
                language=language,
                chunk_type="semantic",
                source_query_id=source_query_id,
                metadata={"sentence_count": len(current_sentences)}
            ))
            
        return chunks

    def process_passages_batch(self, passage_list: List[Dict[str, Any]]) -> List[Chunk]:
        """
        Strategy 3: Metadata-Aware Batch Processing.
        Processes a list of passage raw dicts and tags every chunk with language & source metadata.
        """
        chunks = []
        for passage_dict in passage_list:
            chunk = self.passage_chunking(passage_dict)
            chunks.append(chunk)
        return chunks
