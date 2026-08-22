"""
Streamlit Application for Voice-Enabled Multilingual RAG System.
Features mic capture (st.audio_input), text fallback, live transcription,
retrieved passages display with RRF/Dense/Sparse scores and language tags,
grounded answer generation with citations, guardrail refusal state badges,
and per-stage latency metrics.
"""

import os
import time
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.index_build import MultilingualIndexManager
from src.retrieval import HybridRetriever
from src.stt import SarvamSTTClient
from src.generation import GroqGenerator
from src.guardrails import GuardrailsEngine
from src.harness import PipelineHarness

st.set_page_config(
    page_title="Voice-Enabled RAG (Hindi & English)",
    page_icon="🎙️",
    layout="wide"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5, #43A047);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .badge-hi {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-en {
        background-color: #E3F2FD;
        color: #1565C0;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .passage-card {
        border-left: 4px solid #1E88E5;
        background-color: #F8F9FA;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    .refusal-box {
        border-left: 4px solid #E53935;
        background-color: #FFEBEE;
        padding: 14px 18px;
        border-radius: 4px;
        color: #C62828;
        font-weight: 500;
    }
    .latency-pill {
        background-color: #ECEFF1;
        color: #37474F;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline_components():
    """Cache vector indices and local embedding models for sub-second app startup."""
    manager = MultilingualIndexManager()
    if not manager.load_indices():
        manager.build_all_indices()
    
    # Warmup embedding model on cache load
    manager.embedding_model.encode(["warmup query"], normalize_embeddings=True)
    
    retriever = HybridRetriever(manager)
    stt_client = SarvamSTTClient()
    generator = GroqGenerator()
    
    guardrails = GuardrailsEngine(index_manager=manager)
    guardrails.bind_index_manager(manager)
    
    harness = PipelineHarness(
        retriever=retriever,
        stt_client=stt_client,
        generator=generator,
        use_fixtures=False
    )
    harness.bind_guardrails(guardrails)
    return harness, manager

def main():
    st.markdown('<div class="main-header">🎙️ Voice-Enabled Multilingual RAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">HH Goa 2026 Shortlisting Task 2 — Hybrid Retrieval (FAISS + BM25) with Sarvam STT & Groq Generation</div>', unsafe_allow_html=True)

    with st.spinner("Initializing indices and local embedding model..."):
        harness, manager = load_pipeline_components()

    # Sidebar Options
    st.sidebar.header("⚙️ Pipeline Settings")
    input_mode = st.sidebar.radio("Input Source", ["Microphone 🎙️", "Text Query ⌨️"])
    target_lang = st.sidebar.selectbox("Language Preference", ["Hindi (hi)", "English (en)", "Auto-Detect"])
    lang_code = "hi" if "Hindi" in target_lang else ("en" if "English" in target_lang else "hi")

    st.sidebar.divider()
    st.sidebar.markdown("### 📊 Guardrail Thresholds")
    st.sidebar.info(f"""
    - **Input Cosine Threshold**: `{harness.guardrails_engine.input_cos_threshold:.3f}`
    - **Retrieval Confidence Gate**: `{harness.guardrails_engine.retrieval_conf_threshold:.4f}`
    - **Groundedness Threshold**: `{harness.guardrails_engine.groundedness_sim_threshold:.3f}`
    """)

    # Main Input Area
    audio_bytes = None
    text_query = None

    if input_mode == "Microphone 🎙️":
        st.subheader("🎤 Speak Your Query")
        audio_value = st.audio_input("Record audio query")
        if audio_value is not None:
            if "last_audio_file" not in st.session_state or st.session_state.last_audio_file != id(audio_value):
                st.session_state.audio_bytes = audio_value.read()
                st.session_state.last_audio_file = id(audio_value)
            audio_bytes = st.session_state.get("audio_bytes")
    else:
        st.subheader("⌨️ Type Your Query")
        text_query = st.text_input("Enter your question in Hindi or English:", placeholder="जैसे: कॉर्पोरेशन क्या है? or What is a corporation?")

    if st.button("🚀 Process Pipeline", type="primary", use_container_width=True):
        if not audio_bytes and not text_query:
            st.warning("Please record an audio query or type a question to proceed.")
            return

        with st.spinner("Executing Voice-RAG Pipeline..."):
            pipeline_out = harness.execute_pipeline(
                audio_bytes=audio_bytes,
                text_query=text_query,
                forced_language=lang_code
            )

        # 1. Transcript & Language Detection Output
        st.divider()
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### 📝 Transcript\n**\"{pipeline_out.query_text}\"**")
        with col2:
            lang_badge = f'<span class="badge-hi">Hindi (hi)</span>' if pipeline_out.detected_language == "hi" else f'<span class="badge-en">English (en)</span>'
            st.markdown(f"### Language\n{lang_badge}", unsafe_allow_html=True)

        # 2. Refusal State Display
        if pipeline_out.refusal:
            st.markdown(f"""
            <div class="refusal-box">
                🛡️ <b>Pipeline Refusal State Active</b><br/>
                {pipeline_out.refusal_reason}
            </div>
            """, unsafe_allow_html=True)

        # 3. Retrieved Passages
        if pipeline_out.retrieval and pipeline_out.retrieval.passages:
            st.markdown("### 📚 Retrieved Context Passages (Language-Filtered)")
            for p in pipeline_out.retrieval.passages:
                st.markdown(f"""
                <div class="passage-card">
                    <b>Rank {p.rank} — Passage ID: <code>[{p.id}]</code></b> 
                    <span class="latency-pill">RRF Score: {p.rrf_score:.4f}</span> 
                    <span class="latency-pill">Language: {p.language.upper()}</span><br/>
                    <p style="margin-top:6px; margin-bottom:0;">{p.text}</p>
                </div>
                """, unsafe_allow_html=True)

        # 4. Grounded Answer
        if pipeline_out.generation and not pipeline_out.refusal:
            st.markdown("### 🤖 Grounded Answer")
            st.success(pipeline_out.generation.answer)
            st.caption(f"Citations: {', '.join(pipeline_out.generation.citations)}")

        # 5. Latency Metrics Breakdown
        st.divider()
        st.markdown("### ⏱️ Per-Stage Latency Breakdown")
        m1, m2, m3, m4 = st.columns(4)
        
        stt_ms = pipeline_out.stage_latencies.get("stt_ms", 0.0)
        ret_ms = pipeline_out.stage_latencies.get("retrieval_ms", 0.0)
        gen_ms = pipeline_out.stage_latencies.get("generation_ms", 0.0)
        total_ms = pipeline_out.stage_latencies.get("total_e2e_ms", 0.0)
        
        if total_ms == 0.0:
            total_ms = stt_ms + ret_ms + gen_ms

        m1.metric("STT (Sarvam)", f"{stt_ms:.1f} ms")
        m2.metric("Retrieval (FAISS+BM25)", f"{ret_ms:.1f} ms", help="Target: < 200 ms")
        m3.metric("Generation (Groq)", f"{gen_ms:.1f} ms")
        m4.metric("Total End-to-End", f"{total_ms:.1f} ms")

if __name__ == "__main__":
    main()
