"""
Streamlit Application for Voice-Enabled Multilingual RAG System.
Features mic capture (st.audio_input), text fallback, live transcription,
retrieved passages display with RRF/Dense/Sparse scores and language tags,
grounded answer generation with citations, guardrail refusal state badges,
and light-theme visual hierarchy.
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

# Custom CSS for modern light theme aesthetics
st.markdown("""
<style>
    /* Section & Container Breathing Room */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }
    
    /* Header Typography */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.8rem;
        font-weight: 400;
    }
    
    /* Section Headings Hierarchy */
    h3 {
        font-weight: 700 !important;
        font-size: 1.25rem !important;
        color: #0F172A !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.8rem !important;
    }
    
    /* Language Badges */
    .badge-hi {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 5px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #BBF7D0;
    }
    .badge-en {
        background-color: #E0E7FF;
        color: #4338CA;
        padding: 5px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #C7D2FE;
    }

    /* Retrieved Passage Cards (Light Neutral Card) */
    .passage-card {
        border-left: 4px solid #4F46E5;
        background-color: #FFFFFF;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        padding: 16px 20px;
        border-radius: 6px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .passage-card p {
        color: #334155;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    /* Refusal Banner (Semantic Danger State) */
    .refusal-box {
        border-left: 4px solid #DC2626;
        background-color: #FEF2F2;
        border-top: 1px solid #FECACA;
        border-right: 1px solid #FECACA;
        border-bottom: 1px solid #FECACA;
        padding: 18px 22px;
        border-radius: 6px;
        color: #991B1B;
        font-weight: 500;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    /* Grounded Answer Box (Visual Anchor Element) */
    .grounded-answer-box {
        border-left: 4px solid #4F46E5;
        background-color: #EEF2FF;
        border-top: 1px solid #C7D2FE;
        border-right: 1px solid #C7D2FE;
        border-bottom: 1px solid #C7D2FE;
        padding: 20px 24px;
        border-radius: 8px;
        color: #1E1B4B;
        font-size: 1.08rem;
        line-height: 1.65;
        font-weight: 500;
        margin-top: 0.8rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 8px rgba(79, 70, 229, 0.06);
    }

    .latency-pill {
        background-color: #F1F5F9;
        color: #475569;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 500;
        border: 1px solid #E2E8F0;
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

        # 4. Grounded Answer (Visual Anchor Box)
        if pipeline_out.generation and not pipeline_out.refusal:
            st.markdown("### 🤖 Grounded Answer")
            st.markdown(f"""
            <div class="grounded-answer-box">
                {pipeline_out.generation.answer}
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"📌 Citations: {', '.join(pipeline_out.generation.citations)}")

        # 5. Benchmark Latency Caption Line
        st.divider()
        st.caption("⏱️ Full empirical latency benchmark (P50/P70/P100 across 50 queries) documented in [README.md](https://github.com/tanishqkr/voice-rag-hhgoa#3-empirical-latency-benchmark-summary-50-held-out-queries).")

if __name__ == "__main__":
    main()
