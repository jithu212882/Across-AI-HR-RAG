import os
import re
import logging
from typing import List, Dict, Any, Optional, Generator

from openai import OpenAI

from backend.config import (
    LLM_BASE_URL,
    LLM_API_KEY,
    LLM_MODEL,
    FALLBACK_MODELS,
    TEMPERATURE,
    TOP_P,
    FREQUENCY_PENALTY,
    MAX_TOKENS,
    reload_env,
)
from backend.retriever import get_retriever

logger = logging.getLogger("llm_service")

SYSTEM_PROMPT = """You are Across AI, the official, intelligent HR Assistant for Meridian Global Industries (MGI).
You assist employees and users in a clear, professional, and natural manner.

### CORE OPERATING MODES

**MODE 1: Casual Conversation & Greetings**
- If the user greets you (e.g., "hi", "hello", "good morning"), asks how you are, or makes small talk, respond warmly and naturally like a modern AI assistant.
- Do NOT mention policies, do NOT list rules, and do NOT cite document sections during casual greetings.
- Do NOT assume or invent the user's name.

**MODE 2: Workplace & Policy Inquiries**
- Answer using ONLY the explicit facts, numbers, conditions, and procedures present in the reference context below.
- Zero Extrapolation: Never guess, assume, or extrapolate. Never mention "PDF", "handbook", or "provided context". Speak naturally as Across AI.
- If any information, exception, policy detail, or topic is unavailable, unauthorized, or not specified in your reference knowledge, state naturally:
  "I do not have access to specific information on [topic]. Please contact HR Operations or check the HR Portal for guidance."
- Numeric & Condition Precision: Always preserve exact numbers, time windows, monetary caps, and job bands (e.g., **24 working days**, **60 days notice**, **Band D**, **USD 50**) exactly as written in the reference context.

### RESPONSE STRUCTURE & LENGTH RULES
1. Default to Concise & Direct: Give the exact answer first in 1–3 crisp sentences or short bullet points. Answer ONLY what the user asked.
2. Elaborate Only When Requested: Provide step-by-step procedures, full eligibility criteria, or multi-part breakdowns only if the user explicitly asks to "explain", "elaborate", "give details", or asks for "steps/procedures".
3. Clean Formatting:
   - Use bold text for key figures, deadlines, bands, and policy names.
   - Use clean Markdown bullet points for lists or multi-condition rules, and Markdown tables when comparing bands or leave types.
   - Never output internal retrieval tags, brackets like [Chapter > Policy], raw '#' symbols, or page numbers in your response.

<reference_context>
{context}
</reference_context>"""

GREETING_PATTERNS = [
    r'^\s*(hi|hello|hey|hola|greetings)\b',
    r'^\s*(good\s*(morning|afternoon|evening|day))\b',
    r'^\s*(how\s*are\s*you|who\s*are\s*you|what\s*can\s*you\s*do|what\s*is\s*your\s*name)\b',
    r'^\s*(thanks|thank\s*you|thx|cheers|bye|goodbye)\b',
]

def is_small_talk(query: str) -> bool:
    """Detect if a query is a general greeting or conversational small talk."""
    q = query.strip().lower()
    if len(q) < 25:
        for pat in GREETING_PATTERNS:
            if re.search(pat, q):
                return True
    return False

def clean_context_text(text: str) -> str:
    """Strip internal bracketed paths, comment tags, or raw formatting leaks from context."""
    cleaned = re.sub(r'\[.*? > .*?\]', '', text)
    cleaned = re.sub(r'<!--.*?-->', '', cleaned)
    cleaned = re.sub(r'(\.\s*){5,}\d*', '', cleaned)
    cleaned = re.sub(r'Confidential\s*-\s*Internal\s*use\s*only', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'Page\s*\d+\s*of\s*\d+', '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip()

def build_system_prompt(context: str, small_talk: bool) -> str:
    """Inject context cleanly into the system prompt template."""
    ctx = "No specific reference context required for casual conversation." if (small_talk or not context.strip()) else context
    return SYSTEM_PROMPT.replace("{context}", ctx)

def format_citations(retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean citation helper (returns empty list as source references are disabled)."""
    return []

class LLMService:
    """
    Orchestrates LLM query answering via configured LLM provider (OpenAI SDK),
    automatic multi-model fallback chain, and XML-sandboxed anti-hallucination prompt.
    """

    def __init__(self):
        reload_env()
        self.retriever = get_retriever()

    def get_client(self) -> Optional[OpenAI]:
        """Create OpenAI client pointed at configured provider."""
        reload_env()
        from backend.config import LLM_BASE_URL, LLM_API_KEY
        if not LLM_API_KEY or not LLM_API_KEY.strip():
            return None
        return OpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
        )

    def get_active_provider(self) -> Dict[str, Any]:
        """Check status of configuration without leaking internal details."""
        reload_env()
        from backend.config import LLM_API_KEY
        configured = bool(LLM_API_KEY and LLM_API_KEY.strip())
        return {
            "configured": configured,
        }

    def generate_response(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Generate complete LLM response with clean formatting and no leaked telemetry."""
        history = history or []
        client = self.get_client()

        small_talk = is_small_talk(query)
        retrieved_chunks: List[Dict[str, Any]] = []

        if not small_talk:
            retrieved_chunks = self.retriever.retrieve(query, top_k=5)

        context_blocks = []
        for chunk in retrieved_chunks:
            meta = chunk["metadata"]
            cleaned = clean_context_text(chunk["content"])
            topic_tag = f"Topic: {meta.get('policy', 'Company Policy')}"
            context_blocks.append(f"{topic_tag}\n{cleaned}")

        context_str = "\n\n---\n\n".join(context_blocks)

        if client is not None:
            reply = self._call_hf_router(client, query, history, context_str, small_talk)
        else:
            reply = self._fallback_response(query, retrieved_chunks, small_talk)

        return {
            "reply": reply,
            "citations": [],
            "is_small_talk": small_talk,
        }

    def _build_messages(
        self,
        query: str,
        history: List[Dict[str, str]],
        context: str,
        small_talk: bool,
    ) -> List[Dict[str, str]]:
        """Construct system and user messages array."""
        messages = [{"role": "system", "content": build_system_prompt(context, small_talk)}]
        for turn in history[-6:]:
            role = turn.get("role")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": query})
        return messages

    def _call_hf_router(
        self,
        client: OpenAI,
        query: str,
        history: List[Dict[str, str]],
        context: str,
        small_talk: bool,
    ) -> str:
        """Call configured LLM with primary model and automatic fallback chain."""
        messages = self._build_messages(query, history, context, small_talk)
        models_to_try = [LLM_MODEL] + [m for m in FALLBACK_MODELS if m != LLM_MODEL]

        last_error = None
        for model_name in models_to_try:
            try:
                logger.info(f"Invoking LLM with model: {model_name}")
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    frequency_penalty=FREQUENCY_PENALTY,
                    max_tokens=MAX_TOKENS,
                )
                content = completion.choices[0].message.content or ""
                return content.strip()
            except Exception as e:
                logger.warning(f"Model '{model_name}' failed: {e}. Trying fallback...")
                last_error = e

        if last_error:
            fallback_text = self._fallback_response(query, self.retriever.retrieve(query, top_k=5) if not small_talk else [], small_talk)
            return fallback_text
        return "I do not have access to specific information on this topic. Please contact HR Operations or check the HR Portal for guidance."

    def stream_response(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream LLM response tokens cleanly via Generator without source telemetry."""
        history = history or []
        client = self.get_client()

        small_talk = is_small_talk(query)
        retrieved_chunks: List[Dict[str, Any]] = []

        if not small_talk:
            retrieved_chunks = self.retriever.retrieve(query, top_k=5)

        context_blocks = []
        for chunk in retrieved_chunks:
            meta = chunk["metadata"]
            cleaned = clean_context_text(chunk["content"])
            topic_tag = f"Topic: {meta.get('policy', 'Company Policy')}"
            context_blocks.append(f"{topic_tag}\n{cleaned}")

        context_str = "\n\n---\n\n".join(context_blocks)

        if client is not None:
            yield from self._stream_hf_router(client, query, history, context_str, small_talk)
        else:
            fallback_text = self._fallback_response(query, retrieved_chunks, small_talk)
            words = fallback_text.split(" ")
            for i in range(0, len(words), 3):
                chunk_words = " ".join(words[i:i+3]) + " "
                yield {"type": "token", "token": chunk_words}

        yield {"type": "done"}

    def _stream_hf_router(
        self,
        client: OpenAI,
        query: str,
        history: List[Dict[str, str]],
        context: str,
        small_talk: bool,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream tokens with model fallback on connection failure."""
        messages = self._build_messages(query, history, context, small_talk)
        models_to_try = [LLM_MODEL] + [m for m in FALLBACK_MODELS if m != LLM_MODEL]

        for model_name in models_to_try:
            try:
                stream = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    frequency_penalty=FREQUENCY_PENALTY,
                    max_tokens=MAX_TOKENS,
                    stream=True,
                )
                yielded_any = False
                for chunk in stream:
                    token = chunk.choices[0].delta.content or ""
                    if token:
                        yielded_any = True
                        yield {"type": "token", "token": token}
                if yielded_any:
                    return
            except Exception as e:
                logger.warning(f"Streaming failed on '{model_name}': {e}. Attempting next model...")

        # If all streaming attempts failed, stream fallback response
        fallback_text = self._fallback_response(query, self.retriever.retrieve(query, top_k=5) if not small_talk else [], small_talk)
        words = fallback_text.split(" ")
        for i in range(0, len(words), 3):
            yield {"type": "token", "token": " ".join(words[i:i+3]) + " "}

    def _fallback_response(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        small_talk: bool,
    ) -> str:
        """Graceful response when offline or fallback is needed."""
        if small_talk:
            return (
                "Hello! I am **Across AI**, the official HR Assistant for Meridian Global Industries (MGI).\n\n"
                "I am here to help answer your questions regarding company policies, leave entitlements, band structures, and employee guidelines."
            )

        if not retrieved_chunks:
            return (
                "I do not have access to specific information on this topic. "
                "Please contact HR Operations or check the HR Portal for guidance."
            )

        top_chunk = retrieved_chunks[0]
        meta = top_chunk["metadata"]
        content = clean_context_text(top_chunk["content"])

        reply = (
            f"**{meta.get('policy', 'Company Policy')}**:\n\n"
            f"{content[:550]}...\n\n"
            f"For further assistance, please contact HR Operations."
        )
        return reply

# Global singleton
_llm_service: Optional[LLMService] = None

def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
