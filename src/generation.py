"""
Groq LLM Generation Integration Wrapper.
Includes grounded prompt engineering, tenacity retries, Pydantic contracts,
rate limit response header inspection, and offline fixture mode.
"""

import os
import re
import time
from typing import List, Optional
from pydantic import BaseModel
from groq import Groq, RateLimitError, APIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.retrieval import RetrievedPassage

class GenerationResult(BaseModel):
    answer: str
    citations: List[str]
    generation_latency_ms: float
    is_mock: bool = False
    rate_limit_remaining_requests: Optional[int] = None
    rate_limit_reset_seconds: Optional[float] = None

GROUNDED_PROMPT_TEMPLATE = """You are a grounded QA assistant. Your job is to answer the user's query using ONLY the provided contexts.

Rule 1: Answer strictly based on the provided context passages below. If the context does not contain enough information to answer the question, state: "I cannot answer this question based on the provided dataset."
Rule 2: Cite the context source passage IDs (e.g. [hi_p_0] or [en_p_1]) inline where relevant.
Rule 3: Keep the answer concise and direct (2-4 sentences max). Write the answer in the same language as the user query ({language}). Do NOT output any thinking, reasoning, or meta-commentary. Output ONLY the answer text.

Context Passages:
{formatted_contexts}

User Query ({language}): {query_text}

Grounded Answer:"""

def parse_reset_time_to_seconds(reset_str: str) -> float:
    if not reset_str:
        return 1.0
    reset_str = reset_str.strip()
    total = 0.0
    if 'ms' in reset_str:
        ms_match = re.search(r'(\d+(?:\.\d+)?)ms', reset_str)
        if ms_match:
            total += float(ms_match.group(1)) / 1000.0
    else:
        m_match = re.search(r'(\d+(?:\.\d+)?)m', reset_str)
        s_match = re.search(r'(\d+(?:\.\d+)?)s', reset_str)
        if m_match:
            total += float(m_match.group(1)) * 60.0
        if s_match:
            total += float(s_match.group(1))
    return total if total > 0 else 1.0

class GroqGenerator:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2.0, min=2.0, max=15.0),
        retry=retry_if_exception_type((RateLimitError, APIError, Exception)),
        reraise=True
    )
    def generate_answer(
        self,
        query_text: str,
        retrieved_passages: List[RetrievedPassage],
        language: str = "hi"
    ) -> GenerationResult:
        """Generates grounded answer with inline citations using Groq API."""
        if not self.api_key or self.api_key == "your_groq_api_key_here":
            raise ValueError("GROQ_API_KEY is not configured.")

        t0 = time.time()
        client = Groq(api_key=self.api_key)

        # Format context passages with explicit IDs for citation
        formatted_list = []
        citations = []
        for p in retrieved_passages:
            formatted_list.append(f"[{p.id}] {p.text}")
            citations.append(p.id)

        formatted_contexts = "\n".join(formatted_list)
        lang_str = "Hindi" if language.lower().startswith("hi") else "English"

        prompt = GROUNDED_PROMPT_TEMPLATE.format(
            formatted_contexts=formatted_contexts,
            query_text=query_text,
            language=lang_str
        )

        raw_resp = client.chat.completions.with_raw_response.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a concise factual QA model. Output ONLY the final answer with citations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=256,
            temperature=0.1
        )

        response = raw_resp.parse()
        headers = raw_resp.headers

        # Parse rate limit headers
        remaining_reqs = None
        if "x-ratelimit-remaining-requests" in headers:
            try:
                remaining_reqs = int(headers["x-ratelimit-remaining-requests"])
            except (ValueError, TypeError):
                pass

        reset_sec = None
        if "x-ratelimit-reset-requests" in headers:
            try:
                reset_sec = parse_reset_time_to_seconds(headers["x-ratelimit-reset-requests"])
            except Exception:
                pass

        raw_answer = response.choices[0].message.content.strip()
        
        # Clean potential reasoning tags
        if "<think>" in raw_answer:
            if "</think>" in raw_answer:
                answer = raw_answer.split("</think>")[-1].strip()
            else:
                answer = re.sub(r'<think>.*', '', raw_answer, flags=re.DOTALL).strip()
        else:
            answer = raw_answer

        if not answer:
            answer = raw_answer

        latency_ms = (time.time() - t0) * 1000.0

        return GenerationResult(
            answer=answer,
            citations=citations,
            generation_latency_ms=latency_ms,
            is_mock=False,
            rate_limit_remaining_requests=remaining_reqs,
            rate_limit_reset_seconds=reset_sec
        )

    def generate_fixture(
        self,
        query_text: str,
        retrieved_passages: List[RetrievedPassage],
        language: str = "hi"
    ) -> GenerationResult:
        """Returns deterministic offline grounded response without external API calls."""
        citations = [p.id for p in retrieved_passages]
        first_text = retrieved_passages[0].text if retrieved_passages else "No context"
        mock_answer = f"Grounded response for query '{query_text}' based on context [{retrieved_passages[0].id if retrieved_passages else 'none'}]: {first_text[:100]}..."
        
        return GenerationResult(
            answer=mock_answer,
            citations=citations,
            generation_latency_ms=45.0,
            is_mock=True
        )
