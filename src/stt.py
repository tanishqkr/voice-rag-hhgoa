"""
Sarvam Speech-to-Text (STT) Integration Wrapper.
Includes tenacity exponential backoff retries, Pydantic contracts,
and offline fixture mode for network-free testing.
"""

import os
import time
import requests
from typing import Optional, Dict, Any, Tuple
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

def detect_audio_format(audio_bytes: bytes, fallback_name: str = "audio.wav") -> Tuple[str, str]:
    """Detects MIME type and extension from audio binary header."""
    if audio_bytes.startswith(b'\x1a\x45\xdf\xa3'):
        return "audio/webm", "audio.webm"
    elif audio_bytes.startswith(b'OggS'):
        return "audio/ogg", "audio.ogg"
    elif audio_bytes.startswith(b'RIFF'):
        return "audio/wav", "audio.wav"
    elif audio_bytes.startswith(b'ID3') or audio_bytes.startswith(b'\xff\xfb') or audio_bytes.startswith(b'\xff\xf2'):
        return "audio/mpeg", "audio.mp3"
    elif audio_bytes.startswith(b'ftyp'):
        return "audio/mp4", "audio.m4a"
    else:
        ext = fallback_name.split(".")[-1] if "." in fallback_name else "wav"
        return f"audio/{ext}", fallback_name

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
        
        # Guard against zero-length or ultra-short audio recordings
        if not audio_bytes or len(audio_bytes) < 500:
            return TranscriptionResult(
                text="[Short or empty recording]",
                detected_language="hi",
                confidence=0.0,
                stt_latency_ms=0.1,
                is_mock=True
            )

        mime_type, resolved_filename = detect_audio_format(audio_bytes, fallback_name=file_name)

        headers = {
            "api-subscription-key": api_key
        }
        files = {
            "file": (resolved_filename, audio_bytes, mime_type)
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
        
        # Handle client errors gracefully without crashing the app
        if response.status_code >= 400:
            err_msg = response.text
            if "duration is 0" in err_msg.lower() or "too short" in err_msg.lower():
                return TranscriptionResult(
                    text="[Audio clip too short - please speak for at least 1-2 seconds]",
                    detected_language="hi",
                    confidence=0.0,
                    stt_latency_ms=(time.time() - t0) * 1000.0,
                    is_mock=True
                )
            raise requests.exceptions.HTTPError(
                f"Sarvam STT Error ({response.status_code}): {err_msg}",
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
