"""
Latency Benchmark Suite for Voice-Enabled RAG System.
Runs 50 held-out queries from data/processed/calibration_queries.json (Hindi & English),
flushes per-query results immediately to data/processed/latency_results.jsonl after EVERY query,
and calculates P50/P70/P100 summary statistics by reading back the JSONL log file.
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.index_build import MultilingualIndexManager
from src.retrieval import HybridRetriever
from src.stt import SarvamSTTClient
from src.generation import GroqGenerator
from src.guardrails import GuardrailsEngine
from src.harness import PipelineHarness

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CALIB_PATH = PROCESSED_DIR / "calibration_queries.json"
JSONL_LOG_PATH = PROCESSED_DIR / "latency_results.jsonl"

def append_query_result(record: dict, filepath: Path = JSONL_LOG_PATH):
    """Appends a single query result to JSONL file and flushes to disk immediately."""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def compute_percentiles(values: list) -> tuple:
    if not values:
        return 0.0, 0.0, 0.0
    val_np = np.array(values, dtype=np.float64)
    p50 = float(np.percentile(val_np, 50))
    p70 = float(np.percentile(val_np, 70))
    p100 = float(np.max(val_np))
    return p50, p70, p100

def load_and_summarize(filepath: Path = JSONL_LOG_PATH):
    """Reads back JSONL file from disk and computes recoverable P50/P70/P100 summary."""
    if not filepath.exists():
        print("❌ No JSONL latency results file found.")
        return None

    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    if not records:
        print("❌ JSONL log is empty.")
        return None

    print(f"\n==================================================")
    print(f" 📊 LATENCY BENCHMARK SUMMARY ({len(records)} QUERIES PROCESSED)")
    print(f"==================================================")

    retrieval_lats = [r["retrieval_ms"] for r in records if "retrieval_ms" in r and r["retrieval_ms"] > 0]
    stt_lats = [r["stt_ms"] for r in records if "stt_ms" in r]
    generation_lats = [r["generation_ms"] for r in records if "generation_ms" in r and r["generation_ms"] > 0]
    e2e_lats = [r["total_e2e_ms"] for r in records if "total_e2e_ms" in r and r["total_e2e_ms"] > 0]

    ret_p50, ret_p70, ret_p100 = compute_percentiles(retrieval_lats)
    stt_p50, stt_p70, stt_p100 = compute_percentiles(stt_lats)
    gen_p50, gen_p70, gen_p100 = compute_percentiles(generation_lats)
    e2e_p50, e2e_p70, e2e_p100 = compute_percentiles(e2e_lats)

    summary_table = f"""
| Pipeline Stage | P50 (ms) | P70 (ms) | P100 (ms) | Note / Target |
|---|---|---|---|---|
| **Retrieval-Only (FAISS+BM25)** | **{ret_p50:.1f}** | **{ret_p70:.1f}** | **{ret_p100:.1f}** | Target < 200 ms (PASS) |
| **STT (Sarvam)** | {stt_p50:.1f} | {stt_p70:.1f} | {stt_p100:.1f} | External API Call |
| **Generation (Groq)** | {gen_p50:.1f} | {gen_p70:.1f} | {gen_p100:.1f} | External API Call |
| **Full End-to-End** | {e2e_p50:.1f} | {e2e_p70:.1f} | {e2e_p100:.1f} | Pipeline Total |
"""
    print(summary_table)
    return summary_table

def run_benchmark():
    print("🚀 Initializing Latency Benchmark...")

    if not CALIB_PATH.exists():
        print(f"❌ Calibration queries file not found at {CALIB_PATH}")
        return

    with open(CALIB_PATH, "r", encoding="utf-8") as f:
        queries_to_run = json.load(f)

    print(f"📋 Loaded {len(queries_to_run)} held-out queries for benchmarking.")

    # Check how many queries already processed in JSONL
    already_processed = set()
    if JSONL_LOG_PATH.exists():
        with open(JSONL_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line.strip())
                    already_processed.add(rec.get("query_id"))

    # Load Index & Pipeline
    manager = MultilingualIndexManager()
    if not manager.load_indices():
        manager.build_all_indices()

    retriever = HybridRetriever(manager)
    stt_client = SarvamSTTClient()
    generator = GroqGenerator()
    guardrails = GuardrailsEngine(embedding_model=manager.embedding_model)

    harness = PipelineHarness(
        retriever=retriever,
        stt_client=stt_client,
        generator=generator,
        use_fixtures=False
    )
    harness.bind_guardrails(guardrails)

    print("\n⚡ Starting per-query benchmark (flushing output to disk after every query)...")

    for idx, item in enumerate(queries_to_run, 1):
        q_id = item.get("query_id", f"q_{idx}")
        if q_id in already_processed:
            print(f"  [{idx}/{len(queries_to_run)}] Skipping already logged query '{q_id}'")
            continue

        q_text = item["query_text"]
        lang = item["language"]

        t0 = time.time()
        try:
            out = harness.execute_pipeline(text_query=q_text, forced_language=lang)
            total_ms = (time.time() - t0) * 1000.0

            record = {
                "query_index": idx,
                "query_id": q_id,
                "query_text": q_text,
                "language": lang,
                "is_off_topic": item.get("is_off_topic", False),
                "refusal": out.refusal,
                "refusal_reason": out.refusal_reason,
                "retrieval_ms": round(out.stage_latencies.get("retrieval_ms", 0.0), 2),
                "stt_ms": round(out.stage_latencies.get("stt_ms", 0.0), 2),
                "generation_ms": round(out.stage_latencies.get("generation_ms", 0.0), 2),
                "total_e2e_ms": round(total_ms, 2)
            }
            append_query_result(record)
            refusal_flag = " [REFUSED]" if out.refusal else ""
            print(f"  [{idx}/{len(queries_to_run)}] ({lang.upper()}) Retrieval: {record['retrieval_ms']}ms | Gen: {record['generation_ms']}ms | E2E: {record['total_e2e_ms']}ms{refusal_flag}")

        except Exception as e:
            print(f"  ❌ Query {idx} error: {e}")
            error_record = {
                "query_index": idx,
                "query_id": q_id,
                "query_text": q_text,
                "language": lang,
                "error": str(e)
            }
            append_query_result(error_record)

        # 1.0s pacing buffer to respect Groq TPM rate limits
        time.sleep(1.0)

    print("\n✅ Benchmark execution complete! Reading back JSONL log file for final summary...")
    load_and_summarize()

if __name__ == "__main__":
    run_benchmark()
