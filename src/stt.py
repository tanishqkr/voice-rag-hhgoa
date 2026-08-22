"""
Sarvam Speech-to-Text (STT) Integration Wrapper.
Includes tenacity exponential backoff retries, Pydantic contracts,
and offline fixture mode for network-free testing.
"""

import os
import time
import requests
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

load_dotenv()

class TranscriptionResult(BaseModel):
    text: str
    detected_language: str  # "hi" or "en"
    confidence: float = 1.0
    stt_latency_ms: float = 0.0
    is_mock: bool = False

class SarvamSTTClient:
    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        self.api_key = api_key or os.environ.get("SARVAM_API_KEY")
        self.api_url = "https://api.sarvam.ai/speech-to-text"

    def _get_active_api_key(self) -> str:
        key = self.api_key or os.environ.get("SARVAM_API_KEY")
        if not key or key == "your_sarvam_api_key_here":
            load_dotenv()
            key = os.environ.get("SARVAM_API_KEY")
        return key

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=3.0),
        retry=retry_if_exception_type((requests.exceptions.RequestException, TimeoutError)),
        reraise=True
    )
    def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        file_name: str = "audio.wav",
        language_code: str = "unknown"
    ) -> TranscriptionResult:
        """Transcribes audio bytes using Sarvam STT API (saarika:v2.5) with retries."""
        api_key = self._get_active_api_key()
        if not api_key or api_key == "your_sarvam_api_key_here":
            raise ValueError("SARVAM_API_KEY is not configured in .env file.")

        t0 = time.time()
        headers = {
            "api-subscription-key": api_key
        }
        files = {
            "file": (file_name, audio_bytes, "audio/wav")
        }

        # Map language code to Sarvam format
        if language_code.lower().startswith("hi"):
            sarvam_lang = "hi-IN"
        elif language_code.lower().startswith("en"):
            sarvam_lang = "en-IN"
        else:
            sarvam_lang = "hi-IN"

        data = {
            "model": "saarika:v2.5",
            "language_code": sarvam_lang
        }

        response = requests.post(self.api_url, headers=headers, files=files, data=data, timeout=15)
        
        # Raise clear HTTP error with body context
        if response.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"Sarvam API Error ({response.status_code}): {response.text}",
                response=response
            )

        res_json = response.json()
        latency_ms = (time.time() - t0) * 1000.0

        transcript = res_json.get("transcript", "").strip()
        detected_lang_raw = res_json.get("language_code", sarvam_lang)
        
        # Standardize detected language to "hi" or "en"
        detected_lang = "hi" if "hi" in detected_lang_raw.lower() else "en"

        return TranscriptionResult(
            text=transcript,
            detected_language=detected_lang,
            confidence=float(res_json.get("confidence", 0.95)),
            stt_latency_ms=latency_ms,
            is_mock=False
        )

    def transcribe_fixture(self, mock_text: str, mock_language: str = "hi") -> TranscriptionResult:
        """Returns a deterministic offline fixture result without calling external API."""
        return TranscriptionResult(
            text=mock_text,
            detected_language=mock_language,
            confidence=0.99,
            stt_latency_ms=12.5,
            is_mock=True
        )
