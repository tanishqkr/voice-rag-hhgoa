"""
Pipeline Harness & API Wrappers for Multilingual Voice RAG.
Orchestrates STT -> Language Filtering -> Hybrid Retrieval -> Guardrails -> LLM Generation.
Provides Pydantic boundary contracts, latency logging, and fixture mode.
"""

import time
import uuid
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from src.stt import SarvamSTTClient, TranscriptionResult
from src.retrieval import HybridRetriever, RetrievalResult, RetrievedPassage
from src.generation import GroqGenerator, GenerationResult
from src.guardrails import GuardrailsEngine, GuardrailVerdict

# Structured JSON Logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("voice_rag_harness")

def log_stage_latency(stage_name: str):
    """Decorator to log structured JSON timing events per pipeline stage."""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            t0 = time.time()
            status = "SUCCESS"
            error_msg = None
            try:
                result = func(self, *args, **kwargs)
                return result
            except Exception as e:
                status = "FAILED"
                error_msg = str(e)
                raise e
            finally:
                latency_ms = (time.time() - t0) * 1000.0
                event = {
                    "run_id": getattr(self, "current_run_id", "unknown"),
                    "stage": stage_name,
                    "latency_ms": round(latency_ms, 2),
                    "status": status
                }
                if error_msg:
                    event["error"] = error_msg
                logger.info({"event": event})
        return wrapper
    return decorator

class PipelineOutput(BaseModel):
    query_text: str
    detected_language: str
    run_id: str = "unknown"
    transcription: Optional[TranscriptionResult] = None
    retrieval: Optional[RetrievalResult] = None
    generation: Optional[GenerationResult] = None
    guardrail_status: Dict[str, GuardrailVerdict] = Field(default_factory=dict)
    refusal: bool = False
    refusal_reason: Optional[str] = None
    stage_latencies: Dict[str, float] = Field(default_factory=dict)

class PipelineHarness:
    def __init__(
        self,
        retriever: HybridRetriever,
        stt_client: Optional[SarvamSTTClient] = None,
        generator: Optional[GroqGenerator] = None,
        guardrails_engine: Optional[GuardrailsEngine] = None,
        use_fixtures: bool = False
    ):
        self.retriever = retriever
        self.stt_client = stt_client or SarvamSTTClient()
        self.generator = generator or GroqGenerator()
        self.guardrails_engine = guardrails_engine
        self.use_fixtures = use_fixtures
        self.current_run_id = "unknown"

    def bind_guardrails(self, guardrails_engine: GuardrailsEngine):
        """Binds an empirically calibrated GuardrailsEngine to the pipeline."""
        self.guardrails_engine = guardrails_engine

    @log_stage_latency("stt_stage")
    def run_stt_stage(self, audio_bytes: Optional[bytes] = None, language: str = "hi") -> TranscriptionResult:
        if self.use_fixtures:
            return self.stt_client.transcribe_fixture(mock_text="कॉर्पोरेशन क्या है?", mock_language=language)
        elif self.use_fixtures or not audio_bytes:
            return self.stt_client.transcribe_fixture(mock_text="कॉर्पोरेशन क्या है?", mock_language=language)
        else:
            return self.stt_client.transcribe_audio_bytes(audio_bytes=audio_bytes, language_code=language)

    @log_stage_latency("retrieval_stage")
    def run_retrieval_stage(self, query_text: str, language: str) -> RetrievalResult:
        return self.retriever.retrieve(query_text=query_text, detected_language=language, top_n=5)

    @log_stage_latency("generation_stage")
    def run_generation_stage(
        self,
        query_text: str,
        passages: List[RetrievedPassage],
        language: str
    ) -> GenerationResult:
        if self.use_fixtures:
            return self.generator.generate_fixture(query_text=query_text, retrieved_passages=passages, language=language)
        else:
            return self.generator.generate_answer(query_text=query_text, retrieved_passages=passages, language=language)

    def execute_pipeline(
        self,
        audio_bytes: Optional[bytes] = None,
        text_query: Optional[str] = None,
        forced_language: str = "hi"
    ) -> PipelineOutput:
        """Executes full end-to-end Voice RAG pipeline with stage logging and guardrail checkpoints."""
        t_e2e_0 = time.time()
        self.current_run_id = uuid.uuid4().hex[:8]
        logger.info({"event": {"run_id": self.current_run_id, "stage": "pipeline_start"}})

        stage_latencies = {}

        # 1. Speech-to-Text Stage
        t_stt_0 = time.time()
        if text_query:
            transcription = self.stt_client.transcribe_fixture(mock_text=text_query, mock_language=forced_language)
        else:
            transcription = self.run_stt_stage(audio_bytes=audio_bytes, language=forced_language)
        stage_latencies["stt_ms"] = round((time.time() - t_stt_0) * 1000.0, 2)

        query_text = transcription.text
        language = transcription.detected_language

        output = PipelineOutput(
            run_id=self.current_run_id,
            query_text=query_text,
            detected_language=language,
            transcription=transcription
        )

        # 2. Input Guardrail Checkpoint
        if self.guardrails_engine:
            v_input = self.guardrails_engine.check_input_guardrail(query_text, language)
            output.guardrail_status["input_guardrail"] = v_input
            if not v_input.passed:
                output.refusal = True
                output.refusal_reason = f"Refused by Input Guardrail: {v_input.reason}"
                stage_latencies["total_e2e_ms"] = round((time.time() - t_e2e_0) * 1000.0, 2)
                output.stage_latencies = stage_latencies
                return output

        # 3. Retrieval Stage
        t_ret_0 = time.time()
        retrieval_res = self.run_retrieval_stage(query_text=query_text, language=language)
        stage_latencies["retrieval_ms"] = round(retrieval_res.retrieval_latency_ms, 2)
        output.retrieval = retrieval_res

        # 4. Retrieval Confidence Gate Checkpoint
        if self.guardrails_engine:
            v_conf = self.guardrails_engine.check_retrieval_confidence(retrieval_res.top_score)
            output.guardrail_status["retrieval_confidence"] = v_conf
            if not v_conf.passed:
                output.refusal = True
                output.refusal_reason = f"Refused by Retrieval Confidence Gate: {v_conf.reason}"
                stage_latencies["total_e2e_ms"] = round((time.time() - t_e2e_0) * 1000.0, 2)
                output.stage_latencies = stage_latencies
                return output

        # 5. LLM Generation Stage
        t_gen_0 = time.time()
        generation_res = self.run_generation_stage(query_text=query_text, passages=retrieval_res.passages, language=language)
        stage_latencies["generation_ms"] = round(generation_res.generation_latency_ms, 2)
        output.generation = generation_res

        # 6. Output Groundedness Checkpoint
        if self.guardrails_engine:
            v_ground = self.guardrails_engine.check_output_groundedness(
                query_text=query_text,
                generated_answer=generation_res.answer,
                retrieved_passages=retrieval_res.passages
            )
            output.guardrail_status["output_groundedness"] = v_ground
            if not v_ground.passed:
                output.refusal = True
                output.refusal_reason = f"Refused by Output Groundedness Guardrail: {v_ground.reason}"
                stage_latencies["total_e2e_ms"] = round((time.time() - t_e2e_0) * 1000.0, 2)
                output.stage_latencies = stage_latencies
                return output

        stage_latencies["total_e2e_ms"] = round((time.time() - t_e2e_0) * 1000.0, 2)
        output.stage_latencies = stage_latencies
        return output
