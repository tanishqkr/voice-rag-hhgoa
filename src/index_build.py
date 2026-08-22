"""
Index Builder Module for FAISS Dense Index and BM25 Sparse Index.
Creates language-partitioned (Hindi & English) indices with local embedding generation.
"""

import os
import json
import time
import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import Dict, List, Any, Tuple
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indices"

MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def tokenize_text(text: str) -> List[str]:
    """Simple multilingual tokenizer splitting on whitespace and punctuation."""
    import re
    tokens = re.findall(r'\w+', text.lower(), re.UNICODE)
    return tokens if tokens else [text.lower()]

class LanguageIndex:
    """Holds FAISS dense index, BM25 sparse index, and passage metadata for one language."""
    def __init__(self, language: str, passages: List[Dict[str, Any]], faiss_index: faiss.Index, bm25_index: BM25Okapi):
        self.language = language
        self.passages = passages
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index

class MultilingualIndexManager:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.embedding_model = None
        self.indices: Dict[str, LanguageIndex] = {}

    def load_embedding_model(self):
        if self.embedding_model is None:
            print(f"⚡ Loading local embedding model: {self.model_name}...")
            t0 = time.time()
            self.embedding_model = SentenceTransformer(self.model_name)
            print(f"✅ Loaded embedding model in {time.time() - t0:.2f}s")

    def build_index_for_language(self, language: str, passages: List[Dict[str, Any]]) -> LanguageIndex:
        self.load_embedding_model()
        print(f"🔨 Building indices for language '{language}' ({len(passages)} passages)...")
        t0 = time.time()

        texts = [p["text"] for p in passages]

        # 1. Dense Index (FAISS Inner Product with L2 normalized vectors)
        embeddings = self.embedding_model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        embeddings_np = np.array(embeddings, dtype=np.float32)

        dimension = embeddings_np.shape[1]
        faiss_idx = faiss.IndexFlatIP(dimension)
        faiss_idx.add(embeddings_np)

        # 2. Sparse Index (BM25)
        tokenized_corpus = [tokenize_text(t) for t in texts]
        bm25_idx = BM25Okapi(tokenized_corpus)

        print(f"✅ Indices built for '{language}' in {time.time() - t0:.2f}s (Dim: {dimension}, Total: {len(passages)})")
        lang_idx = LanguageIndex(language=language, passages=passages, faiss_index=faiss_idx, bm25_index=bm25_idx)
        self.indices[language] = lang_idx
        return lang_idx

    def build_all_indices(self, hi_path: Path = None, en_path: Path = None):
        hi_path = hi_path or (PROCESSED_DIR / "hindi_passages.json")
        en_path = en_path or (PROCESSED_DIR / "english_passages.json")

        if hi_path.exists():
            with open(hi_path, "r", encoding="utf-8") as f:
                hi_passages = json.load(f)
            self.build_index_for_language("hi", hi_passages)

        if en_path.exists():
            with open(en_path, "r", encoding="utf-8") as f:
                en_passages = json.load(f)
            self.build_index_for_language("en", en_passages)

    def save_indices(self, index_dir: Path = INDEX_DIR):
        index_dir.mkdir(parents=True, exist_ok=True)
        for lang, lang_idx in self.indices.items():
            lang_path = index_dir / lang
            lang_path.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(lang_idx.faiss_index, str(lang_path / "faiss.index"))
            
            # Save BM25 and passages metadata
            with open(lang_path / "bm25.pkl", "wb") as f:
                pickle.dump(lang_idx.bm25_index, f)
                
            with open(lang_path / "passages.json", "w", encoding="utf-8") as f:
                json.dump(lang_idx.passages, f, ensure_ascii=False)
                
        print(f"💾 Saved all indices to {index_dir}")

    def load_indices(self, index_dir: Path = INDEX_DIR) -> bool:
        self.load_embedding_model()
        if not index_dir.exists():
            return False

        loaded = False
        for lang in ["hi", "en"]:
            lang_path = index_dir / lang
            faiss_file = lang_path / "faiss.index"
            bm25_file = lang_path / "bm25.pkl"
            passages_file = lang_path / "passages.json"

            if faiss_file.exists() and bm25_file.exists() and passages_file.exists():
                print(f"⚡ Loading cached indices for '{lang}'...")
                faiss_idx = faiss.read_index(str(faiss_file))
                with open(bm25_file, "rb") as f:
                    bm25_idx = pickle.load(f)
                with open(passages_file, "r", encoding="utf-8") as f:
                    passages = json.load(f)
                    
                self.indices[lang] = LanguageIndex(lang, passages, faiss_idx, bm25_idx)
                loaded = True

        return loaded
