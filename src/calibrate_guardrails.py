"""
Empirical Guardrail Threshold Calibration Script.
Evaluates 50 held-out calibration queries (in-domain vs off-topic)
against the corpus index using local sentence embeddings,
finds optimal F1 classification thresholds, and saves configuration.
"""

import json
import numpy as np
from pathlib import Path
from src.index_build import MultilingualIndexManager
from src.retrieval import HybridRetriever

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CALIB_PATH = PROCESSED_DIR / "calibration_queries.json"
CONFIG_PATH = PROCESSED_DIR / "guardrail_config.json"

def calibrate():
    print("==================================================")
    print("   Empirical Guardrail Threshold Calibration      ")
    print("==================================================")
    
    if not CALIB_PATH.exists():
        print(f"❌ Calibration file not found at {CALIB_PATH}")
        return

    with open(CALIB_PATH, "r", encoding="utf-8") as f:
        calib_queries = json.load(f)
        
    print(f"📊 Loaded {len(calib_queries)} held-out calibration queries.")

    # Initialize Index Manager & Retriever
    manager = MultilingualIndexManager()
    if not manager.load_indices():
        print("🔨 Building fresh indices for calibration...")
        manager.build_all_indices()

    retriever = HybridRetriever(manager)

    # Pre-compute corpus sample embeddings per language once
    corpus_embs = {}
    for lang, lang_idx in manager.indices.items():
        sample_texts = [p["text"] for p in lang_idx.passages[:300]]
        corpus_embs[lang] = manager.embedding_model.encode(sample_texts, normalize_embeddings=True)

    # 1. Calibrate Input Guardrail Cosine Distance Threshold
    print("\n🎯 1. Calibrating Input Off-Topic Threshold...")
    in_domain_sims = []
    off_topic_sims = []

    for item in calib_queries:
        q_text = item["query_text"]
        lang = item["language"]
        is_offtopic = item["is_off_topic"]

        if lang in corpus_embs:
            q_emb = manager.embedding_model.encode([q_text], normalize_embeddings=True)[0]
            sims = np.dot(corpus_embs[lang], q_emb)
            max_sim = float(np.max(sims))

            if is_offtopic:
                off_topic_sims.append(max_sim)
            else:
                in_domain_sims.append(max_sim)

    print(f"   In-domain max cosine sim: mean={np.mean(in_domain_sims):.3f}, min={np.min(in_domain_sims):.3f}")
    print(f"   Off-topic max cosine sim: mean={np.mean(off_topic_sims):.3f}, max={np.max(off_topic_sims):.3f}")

    # Sweep thresholds to find maximum F1 score
    best_input_threshold = 0.35
    best_f1 = -1.0

    for th in np.arange(0.20, 0.60, 0.01):
        tp = sum(1 for s in in_domain_sims if s >= th)
        fp = sum(1 for s in off_topic_sims if s >= th)
        fn = sum(1 for s in in_domain_sims if s < th)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_input_threshold = float(th)

    print(f"✅ Calibrated Input Cosine Threshold: {best_input_threshold:.3f} (F1: {best_f1:.4f})")

    # 2. Calibrate Retrieval Confidence Gate Threshold
    print("\n🎯 2. Calibrating Retrieval Confidence Gate Threshold...")
    in_domain_rrf = []
    off_topic_rrf = []

    for item in calib_queries:
        res = retriever.retrieve(query_text=item["query_text"], detected_language=item["language"], top_n=5)
        top_rrf = res.top_score

        if item["is_off_topic"]:
            off_topic_rrf.append(top_rrf)
        else:
            in_domain_rrf.append(top_rrf)

    best_conf_threshold = float(np.percentile(in_domain_rrf, 5)) if in_domain_rrf else 0.020
    print(f"   In-domain RRF score range: [{min(in_domain_rrf):.4f}, {max(in_domain_rrf):.4f}]")
    print(f"   Off-topic RRF score range: [{min(off_topic_rrf):.4f}, {max(off_topic_rrf):.4f}]")
    print(f"✅ Calibrated Retrieval Confidence Threshold: {best_conf_threshold:.4f}")

    # 3. Groundedness Similarity Threshold
    best_groundedness_threshold = 0.40

    # Save calibrated configuration
    config_data = {
        "input_cos_threshold": round(best_input_threshold, 3),
        "retrieval_conf_threshold": round(best_conf_threshold, 4),
        "groundedness_sim_threshold": round(best_groundedness_threshold, 3),
        "calibration_samples": len(calib_queries),
        "in_domain_samples": len(in_domain_sims),
        "off_topic_samples": len(off_topic_sims)
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    print(f"\n💾 Saved empirical guardrail configuration to {CONFIG_PATH}")
    print("==================================================")

if __name__ == "__main__":
    calibrate()
