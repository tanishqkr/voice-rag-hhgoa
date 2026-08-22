"""
Final Verification Runner Script.
Runs exactly ONE Hindi query and ONE English query through the full live pipeline
(Real FAISS + BM25 retrieval, Real Guardrails, Real Groq LLM API).
Prints full output and per-stage latency breakdown.
"""

import os
import sys
import json
import time
from dotenv import load_dotenv

# Ensure environment loaded
load_dotenv()

from src.index_build import MultilingualIndexManager
from src.retrieval import HybridRetriever
from src.stt import SarvamSTTClient
from src.generation import GroqGenerator
from src.guardrails import GuardrailsEngine
from src.harness import PipelineHarness

def run_live_verification():
    print("==================================================")
    print("   FULL END-TO-END LIVE PIPELINE VERIFICATION    ")
    print("==================================================")

    # 1. Initialize Index & Retriever
    manager = MultilingualIndexManager()
    if not manager.load_indices():
        print("🔨 Building indices...")
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

    # 2. Test Query 1: Hindi Live Pipeline
    print("\n--------------------------------------------------")
    print("🇮🇳 TEST 1: LIVE HINDI QUERY")
    print("--------------------------------------------------")
    hi_query = "कॉर्पोरेशन क्या है?"
    print(f"Query Text: '{hi_query}' (Language: hi)")

    t0_hi = time.time()
    out_hi = harness.execute_pipeline(text_query=hi_query, forced_language="hi")
    total_hi_ms = (time.time() - t0_hi) * 1000.0

    print(f"\n✅ Transcript: '{out_hi.query_text}' (Detected: {out_hi.detected_language})")
    print(f"   Refusal State: {out_hi.refusal} ({out_hi.refusal_reason if out_hi.refusal else 'None'})")
    
    if out_hi.retrieval and out_hi.retrieval.passages:
        top_p = out_hi.retrieval.passages[0]
        print(f"   Top Retrieved Passage ID: [{top_p.id}] (RRF Score: {top_p.rrf_score:.4f})")
        print(f"   Context Text: '{top_p.text[:120]}...'")

    if out_hi.generation:
        print(f"\n   🤖 Groq Grounded Answer:\n   \"{out_hi.generation.answer}\"")
        print(f"   Citations: {out_hi.generation.citations}")

    print(f"\n⏱️ Per-Stage Latency Breakdown (Hindi):")
    print(f"   - STT Stage: {out_hi.stage_latencies.get('stt_ms', 0):.1f} ms")
    print(f"   - Retrieval Stage (FAISS + BM25): {out_hi.stage_latencies.get('retrieval_ms', 0):.1f} ms (Target <200ms)")
    print(f"   - Generation Stage (Groq): {out_hi.stage_latencies.get('generation_ms', 0):.1f} ms")
    print(f"   - Total End-to-End: {total_hi_ms:.1f} ms")

    # 3. Test Query 2: English Live Pipeline
    print("\n--------------------------------------------------")
    print("🇬🇧 TEST 2: LIVE ENGLISH QUERY")
    print("--------------------------------------------------")
    en_query = "What is a corporation?"
    print(f"Query Text: '{en_query}' (Language: en)")

    t0_en = time.time()
    out_en = harness.execute_pipeline(text_query=en_query, forced_language="en")
    total_en_ms = (time.time() - t0_en) * 1000.0

    print(f"\n✅ Transcript: '{out_en.query_text}' (Detected: {out_en.detected_language})")
    print(f"   Refusal State: {out_en.refusal} ({out_en.refusal_reason if out_en.refusal else 'None'})")
    
    if out_en.retrieval and out_en.retrieval.passages:
        top_p = out_en.retrieval.passages[0]
        print(f"   Top Retrieved Passage ID: [{top_p.id}] (RRF Score: {top_p.rrf_score:.4f})")
        print(f"   Context Text: '{top_p.text[:120]}...'")

    if out_en.generation:
        print(f"\n   🤖 Groq Grounded Answer:\n   \"{out_en.generation.answer}\"")
        print(f"   Citations: {out_en.generation.citations}")

    print(f"\n⏱️ Per-Stage Latency Breakdown (English):")
    print(f"   - STT Stage: {out_en.stage_latencies.get('stt_ms', 0):.1f} ms")
    print(f"   - Retrieval Stage (FAISS + BM25): {out_en.stage_latencies.get('retrieval_ms', 0):.1f} ms (Target <200ms)")
    print(f"   - Generation Stage (Groq): {out_en.stage_latencies.get('generation_ms', 0):.1f} ms")
    print(f"   - Total End-to-End: {total_en_ms:.1f} ms")
    print("==================================================")

if __name__ == "__main__":
    run_live_verification()
