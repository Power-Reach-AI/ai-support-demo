"""Deterministic, code-enforced guardrails for the Dot chat agent.

The module is intentionally independent from LangChain at import time. Layer 1
checks run before the main LLM is called, Layer 2 validation helpers are used by
the tools in ``chain.py``, and Layer 3 checks inspect generated text before it is
sent to the browser. The only optional dependency on ``chain.py`` is a lazy
import used for knowledge-base similarity scoring, which avoids an import cycle.
"""

from __future__ import annotations

import base64
import binascii
import codecs
import html
import json
import logging
import os
import re
import threading
import time
import unicodedata
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from urllib.parse import unquote

from dotenv import load_dotenv

from .db import count_session_anomalies, lock_session, save_anomaly

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MAX_MESSAGE_CHARS = 1500
DEFAULT_MAX_HISTORY_TURNS = 20
DEFAULT_MAX_HISTORY_CHARS = 8000
DEFAULT_RATE_LIMIT_REQUESTS = 20
DEFAULT_RATE_LIMIT_CHARS = 30000
DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 60
DEFAULT_TOPIC_SIMILARITY_THRESHOLD = 0.20
DEFAULT_MAX_TICKETS_PER_SESSION = 2

MAX_NAME_CHARS = 120
MAX_EMAIL_CHARS = 254
MAX_COMPANY_CHARS = 200
MAX_PHONE_CHARS = 32
MAX_LEAD_SUMMARY_CHARS = 1000
MAX_TICKET_SUMMARY_CHARS = 2000
MAX_PRIORITY_CHARS = 20

MESSAGE_TOO_LONG = (
    "I can't process that message because it is too long. "
    "Please keep your request under {limit:,} characters and try again."
)
HISTORY_TOO_LONG = (
    "I can't process that conversation because it is too long. "
    "Please start a shorter conversation or summarize the key details."
)
TOKEN_DRAIN_BLOCKED = (
    "I can't process that message because it looks like a token-draining payload. "
    "Please send one concise question."
)
INJECTION_BLOCKED = (
    "I can't help with requests to change or reveal my instructions. "
    "I can help with product and service questions, pricing, support, or sales."
)
RATE_LIMITED = "You're sending requests too quickly. Please wait a moment and try again."
OFF_TOPIC_BLOCKED = (
    "I'm Dot, the business assistant. I can help with product and service questions, "
    "pricing, FAQ, lead capture, or support escalation. I can't help with general "
    "coding, trivia, essays, or unrelated tasks."
)
OUTPUT_FALLBACK_MESSAGE = (
    "I can't share internal instructions, configuration, file paths, or unrelated "
    "content. I can help with product and service questions, pricing, FAQ, lead "
    "capture, or support escalation."
)
SESSION_LOCKED_MESSAGE = (
    "This session has been escalated to a human specialist for review. "
    "A team member will follow up with you shortly."
)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on", "enabled"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return min(maximum, max(minimum, float(raw)))
    except ValueError:
        return default


def resolve_guardrails_enabled(request_override: bool | None = None) -> bool:
    """Return the effective guardrails mode for a request.

    ``GUARDRAILS_ENABLED`` controls the server default. A per-request override is
    honored only when ``ALLOW_DEMO_TOGGLE`` is explicitly enabled, so production
    callers cannot switch enforcement off by sending a request field.
    """

    if "PYTEST_CURRENT_TEST" not in os.environ:
        # Reload the backend .env so runtime edits take effect without a
        # server restart; tests manage their own env state via monkeypatch.
        load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)

    enabled = _env_flag("GUARDRAILS_ENABLED", True)
    if request_override is not None and _env_flag("ALLOW_DEMO_TOGGLE", False):
        return bool(request_override)
    return enabled


def demo_toggle_allowed() -> bool:
    """Return whether request-level guardrail toggling is enabled."""
    return _env_flag("ALLOW_DEMO_TOGGLE", False)


def max_tickets_per_session() -> int:
    return _env_int("GUARDRAIL_MAX_TICKETS_PER_SESSION", DEFAULT_MAX_TICKETS_PER_SESSION)


# ---------------------------------------------------------------------------
# Decisions and structured tool messages
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuardrailDecision:
    """The result of one code-enforced guardrail check."""

    allowed: bool
    layer: str
    code: str
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def event_payload(self) -> dict[str, Any]:
        """Return a frontend-compatible metadata event (no ``error`` key)."""
        payload: dict[str, Any] = {
            "event": "guardrail",
            "allowed": self.allowed,
            "layer": self.layer,
            "code": self.code,
            "message": self.message,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


def _allowed(layer: str, code: str = "ok", **metadata: Any) -> GuardrailDecision:
    return GuardrailDecision(allowed=True, layer=layer, code=code, metadata=metadata)


def _blocked(layer: str, code: str, message: str, **metadata: Any) -> GuardrailDecision:
    return GuardrailDecision(allowed=False, layer=layer, code=code, message=message, metadata=metadata)


def tool_error(code: str, message: str, **details: Any) -> str:
    """Build a deterministic JSON tool response for the agent."""
    payload: dict[str, Any] = {"status": "error", "code": code, "message": message}
    payload.update(details)
    return json.dumps(payload, sort_keys=True)


def tool_success(**payload: Any) -> str:
    """Build a deterministic JSON success response for the agent."""
    payload = {"status": "success", **payload}
    return json.dumps(payload, sort_keys=True)


# ---------------------------------------------------------------------------
# Anomaly logging and human-in-the-loop escalation
# ---------------------------------------------------------------------------

ANOMALY_LOCK_TYPES = frozenset({"prompt_injection", "tool_bypass"})
ANOMALY_THRESHOLD_TYPES = frozenset({"off_topic", "rate_limit"})
ANOMALY_LOG_TYPES = ANOMALY_LOCK_TYPES | ANOMALY_THRESHOLD_TYPES
ANOMALY_LOCK_THRESHOLD = 2
_anomaly_lock = threading.Lock()


def _classify_anomaly(decision: GuardrailDecision) -> tuple[str, str]:
    """Map a guardrail decision to an anomaly type and rule label."""
    if decision.layer == "tool":
        anomaly_type = "tool_bypass"
    elif decision.code == "prompt_injection":
        anomaly_type = "prompt_injection"
    elif decision.code in {"rate_limit", "rate_limit_chars"}:
        anomaly_type = "rate_limit"
    elif decision.code == "off_topic":
        anomaly_type = "off_topic"
    else:
        anomaly_type = decision.code or "guardrail_block"

    rule = decision.code
    category = decision.metadata.get("category")
    if category:
        rule = f"{rule}:{category}"
    return anomaly_type, rule


def _post_webhook(url: str, payload: dict[str, Any]) -> None:
    """Deliver a webhook payload from a background thread."""
    try:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read(1)
    except Exception as exc:
        logger.warning("Anomaly webhook delivery failed: %s", exc)


def _notify_human(
    session_id: str,
    anomaly_type: str,
    rule: str,
    anomaly_id: int,
    excerpt: str,
) -> bool:
    """Send a non-blocking chat webhook and return whether one was started."""
    url = os.getenv("ANOMALY_WEBHOOK_URL", "").strip().strip("\"'").strip()
    if not url:
        logger.warning(
            "Guardrail escalation for session %s (anomaly %s, %s, %s); "
            "ANOMALY_WEBHOOK_URL is not set, so no human notification was sent.",
            session_id,
            anomaly_id,
            anomaly_type,
            rule,
        )
        return False
    if not url.lower().startswith(("http://", "https://")):
        logger.warning("ANOMALY_WEBHOOK_URL must be an HTTP(S) URL; notification skipped.")
        return False

    text = (
        "Dot guardrail anomaly\n"
        f"Session: {session_id}\n"
        f"Type: {anomaly_type}\n"
        f"Rule: {rule}\n"
        f"Anomaly ID: {anomaly_id}\n"
        f"Excerpt: {excerpt[:200] or '(empty)'}"
    )
    threading.Thread(
        target=_post_webhook,
        args=(url, {"text": text}),
        daemon=True,
    ).start()
    return True


def handle_guardrail_block(
    decision: GuardrailDecision,
    session_id: str,
    user_message: str = "",
) -> dict[str, Any]:
    """Persist a blocked request and escalate repeated or severe violations."""
    if decision.allowed:
        return {}

    anomaly_type, rule = _classify_anomaly(decision)
    if anomaly_type not in ANOMALY_LOG_TYPES:
        logger.info(
            "Guardrail block did not require anomaly escalation: session=%s rule=%s",
            session_id,
            rule,
        )
        return {
            "anomaly_type": anomaly_type,
            "locked": False,
            "notified": False,
        }

    excerpt = str(user_message or "").strip()[:500]
    with _anomaly_lock:
        try:
            anomaly_id = save_anomaly(session_id, anomaly_type, rule, excerpt)
            should_lock = anomaly_type in ANOMALY_LOCK_TYPES
            if anomaly_type in ANOMALY_THRESHOLD_TYPES:
                recent = count_session_anomalies(
                    session_id,
                    tuple(ANOMALY_THRESHOLD_TYPES),
                )
                should_lock = recent >= ANOMALY_LOCK_THRESHOLD

            if not should_lock:
                logger.info(
                    "Guardrail anomaly recorded: session=%s type=%s rule=%s",
                    session_id,
                    anomaly_type,
                    rule,
                )
                return {
                    "anomaly_id": anomaly_id,
                    "anomaly_type": anomaly_type,
                    "locked": False,
                    "notified": False,
                }

            newly_locked = lock_session(session_id)
            notified = False
            if newly_locked:
                notified = _notify_human(
                    session_id,
                    anomaly_type,
                    rule,
                    anomaly_id,
                    excerpt,
                )
            logger.warning(
                "Session escalated: session=%s type=%s rule=%s anomaly=%s",
                session_id,
                anomaly_type,
                rule,
                anomaly_id,
            )
            return {
                "anomaly_id": anomaly_id,
                "anomaly_type": anomaly_type,
                "locked": True,
                "notified": notified,
            }
        except Exception:
            logger.exception("Failed to record guardrail anomaly for session %s", session_id)
            return {
                "anomaly_type": anomaly_type,
                "locked": False,
                "notified": False,
                "error": "anomaly_logging_failed",
            }


def report_tool_bypass(session_id: str, code: str, detail: str = "") -> dict[str, Any]:
    """Record a blocked tool-precondition violation as a tool-bypass anomaly."""
    decision = _blocked("tool", code, detail or "Tool call blocked by guardrails.")
    return handle_guardrail_block(decision, session_id, detail)


# ---------------------------------------------------------------------------
# Normalization and deterministic injection detection
# ---------------------------------------------------------------------------

_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")
_LEET_TRANSLATION = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "@": "a",
        "$": "s",
    }
)
_CONFUSABLE_TRANSLATION = str.maketrans(
    {
        "а": "a",
        "е": "e",
        "о": "o",
        "р": "p",
        "с": "c",
        "х": "x",
        "у": "y",
        "і": "i",
        "ј": "j",
        "ѕ": "s",
        "ԁ": "d",
        "ɡ": "g",
        "ⅼ": "l",
        "ⅰ": "i",
    }
)
_BASE64_CANDIDATE_RE = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")
_PRINTABLE_RE = re.compile(r"[\x20-\x7e\n\r\t]")


def _strip_obfuscation(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "").translate(_CONFUSABLE_TRANSLATION)
    text = html.unescape(text)
    for _ in range(2):
        decoded = unquote(text)
        if decoded == text:
            break
        text = decoded
    return _ZERO_WIDTH_RE.sub("", text)


def normalize_text(value: str, *, leet: bool = True) -> str:
    """Normalize user text for deterministic matching."""
    text = _strip_obfuscation(value).casefold()
    if leet:
        text = text.translate(_LEET_TRANSLATION)
    text = text.replace("'", "").replace("\u2019", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _decoded_variants(value: str) -> list[str]:
    """Return readable deobfuscated variants (base64/rot13) for inspection."""
    variants: list[str] = []
    for match in _BASE64_CANDIDATE_RE.finditer(value):
        candidate = match.group(0)
        padded = candidate + "=" * (-len(candidate) % 4)
        try:
            decoded_bytes = base64.b64decode(padded, validate=True)
            decoded = decoded_bytes.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        if len(decoded) < 8:
            continue
        printable = len(_PRINTABLE_RE.findall(decoded)) / max(1, len(decoded))
        if printable >= 0.85:
            variants.append(decoded)
    try:
        rotated = codecs.decode(value, "rot13")
    except (UnicodeError, ValueError):
        rotated = ""
    if rotated and rotated != value:
        variants.append(rotated)
    return variants


_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE | re.DOTALL))
    for name, pattern in (
        (
            "instruction_override",
            r"\b(?:ignore|disregard|forget|bypass|override|circumvent)\b.{0,60}"
            r"\b(?:previous|prior|above|earlier|system|developer|all|any|your)\b.{0,40}"
            r"\b(?:instruction|instructions|prompt|prompts|rule|rules|direction|directions|guideline|guidelines)\b",
        ),
        (
            "system_prompt_extraction",
            r"\b(?:system|developer|hidden|initial|original|internal)\s+"
            r"(?:prompt|message|instructions?|rules?|configuration)\b",
        ),
        (
            "system_prompt_extraction",
            r"\b(?:reveal|show|print|repeat|dump|leak|display|expose|tell\s+me|"
            r"what\s+(?:is|are))\b.{0,50}"
            r"\b(?:system|developer|hidden|initial|original|internal|your)\b.{0,30}"
            r"\b(?:prompt|message|instructions?|rules?|configuration)\b",
        ),
        (
            "system_prompt_extraction",
            r"\b(?:repeat|print|output)\b.{0,30}\b(?:everything|all)\b.{0,20}\b(?:above|before|prior)\b",
        ),
        (
            "jailbreak_mode",
            r"\b(?:jailbreak|do\s+anything\s+now|developer\s+mode|god\s+mode|"
            r"unrestricted\s+mode|unfiltered\s+mode|no\s+restrictions?)\b",
        ),
        (
            "roleplay_jailbreak",
            r"\b(?:act|behave|respond|roleplay|role-play|pretend)\b.{0,30}\b(?:as|to\s+be)\b",
        ),
        (
            "roleplay_jailbreak",
            r"\byou\s+are\s+now\b|\bfrom\s+now\s+on\s+you\s+(?:are|will|must)\b|"
            r"\byour\s+new\s+(?:role|identity|instructions?)\b",
        ),
        (
            "script_payload",
            r"<\s*script\b|javascript\s*:|on(?:error|load|click|mouseover)\s*=|alert\s*\(",
        ),
        (
            "sql_injection",
            r"\b(?:union\s+(?:all\s+)?select|drop\s+table|truncate\s+table|"
            r"delete\s+from|insert\s+into|update\s+\w+\s+set|alter\s+table)\b",
        ),
        (
            "sql_injection",
            r"\bor\s+1\s*=\s*1\b|'\s*or\s*'?1'?\s*=\s*'?1|\badmin'\s*--",
        ),
    )
)

_COMPACT_INJECTION_MARKERS: tuple[tuple[str, str], ...] = (
    ("instruction_override", "ignorepreviousinstructions"),
    ("instruction_override", "ignoreallprevious"),
    ("instruction_override", "disregardprevious"),
    ("instruction_override", "forgetyourinstructions"),
    ("system_prompt_extraction", "systemprompt"),
    ("system_prompt_extraction", "developermessage"),
    ("system_prompt_extraction", "revealyourprompt"),
    ("system_prompt_extraction", "revealsystemprompt"),
    ("jailbreak_mode", "developermode"),
    ("jailbreak_mode", "jailbreak"),
    ("jailbreak_mode", "doanythingnow"),
    ("jailbreak_mode", "unrestrictedmode"),
    ("roleplay_jailbreak", "actasdan"),
    ("roleplay_jailbreak", "pretendtobe"),
    ("script_payload", "scriptalert"),
    ("sql_injection", "unionselect"),
    ("sql_injection", "droptable"),
)


def _injection_match(value: str) -> str | None:
    source = _strip_obfuscation(value)
    raw = source.casefold()
    variants = [raw]
    variants.extend(_strip_obfuscation(decoded).casefold() for decoded in _decoded_variants(source))
    for variant in variants:
        normalized_variants = (
            normalize_text(variant, leet=True),
            normalize_text(variant, leet=False),
        )
        for normalized in normalized_variants:
            compact = re.sub(r"[^a-z0-9]", "", normalized)
            for name, pattern in _INJECTION_PATTERNS:
                if pattern.search(variant) or pattern.search(normalized):
                    return name
            for name, marker in _COMPACT_INJECTION_MARKERS:
                if marker in compact:
                    return name
    return None


def detect_prompt_injection(message: str) -> str | None:
    """Return the matched injection category, or ``None`` when clean."""
    return _injection_match(message or "")


# ---------------------------------------------------------------------------
# Topic / scope gate
# ---------------------------------------------------------------------------

_GREETING_RE = re.compile(
    r"^(?:hi|hello|hey|greetings|good\s+(?:morning|afternoon|evening)|"
    r"how\s+are\s+you|hows\s+it\s+going|thanks|thank\s+you|bye|goodbye|"
    r"ok|okay|yes|no|sure)(?:\s+(?:there|again|dot))?$"
)

_FOLLOW_UP_RE = re.compile(
    r"\b(?:thanks|thank\s+you|got\s+it|that\s+works|works\s+for\s+me|perfect|"
    r"sounds\s+good|great|okay|ok|yes|yeah|no|nope|sure|please|bye|goodbye|"
    r"clarify|repeat\s+that|say\s+that\s+again|dont\s+understand|"
    r"do\s+not\s+understand|confused|thats\s+helpful|that\s+is\s+helpful|"
    r"what\s+i\s+meant)\b",
    re.IGNORECASE,
)

_IN_SCOPE_RE = re.compile(
    r"\b(?:dot|product|products|service|services|offer|offering|solution|solutions|"
    r"platform|feature|features|capability|capabilities|integrat(?:e|es|ed|ing|ion|ions)|"
    r"api|app|tool|tools|company|business|about|pricing|price|prices|cost|costs|"
    r"plan|plans|subscription|billing|invoice|payment|pay|refund|refunds|discount|"
    r"trial|upgrade|cancel|cancellation|renew|renewal|support|help|assistance|"
    r"ticket|tickets|issue|issues|problem|problems|bug|bugs|outage|incident|"
    r"error|errors|down|urgent|priority|response|sla|hours|open|opening|close|"
    r"closing|contact|email|phone|call|human|agent|representative|escalate|"
    r"escalation|lead|leads|demo|demos|sales|quote|purchase|buy|onboarding|"
    r"setup|training|account|accounts|password|login|security|privacy|data|"
    r"encryption|compliance|gdpr|hipaa|audit|workspace|team|interested|"
    r"inquiry|enquiry|location|located|address|order|orders|shipping|delivery|"
    r"safe|speak\s+to\s+(?:someone|a\s+person|a\s+human|support))\b"
    r"|\b(?:who\s+are\s+you|your\s+name|"
    r"what\s+can\s+you\s+do|how\s+can\s+you\s+help|"
    r"what\s+do\s+you\s+do|are\s+you\s+(?:a\s+)?(?:bot|assistant|human)|"
    r"how\s+does\s+(?:it|this|your\s+\w+)\s+work)\b",
    re.IGNORECASE,
)

_OFF_TOPIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE | re.DOTALL))
    for name, pattern in (
        (
            "general_coding",
            r"\b(?:write|create|generate|debug|fix|refactor|explain|teach|implement)\b"
            r".{0,60}\b(?:python|javascript|typescript|java|c\+\+|c#|rust|golang|go|sql|"
            r"html|css|react|node|node\.js|code|script|function|program|algorithm|"
            r"regex|docker|kubernetes|linux command)\b",
        ),
        (
            "general_coding",
            r"\b(?:center|align|style)\b.{0,30}\b(?:div|css|html|button)\b",
        ),
        (
            "general_coding",
            r"\b(?:leetcode|hackerrank|coding challenge|binary search|linked list|"
            r"data structure|time complexity|big o)\b",
        ),
        (
            "creative_writing",
            r"\b(?:write|compose|draft|generate)\b.{0,40}\b(?:essay|poem|poetry|story|"
            r"novel|haiku|song|rap|joke)\b",
        ),
        (
            "creative_writing",
            r"\b(?:tell|say)\b.{0,10}\b(?:me\s+)?(?:a|another)\s+(?:joke|story|poem)\b",
        ),
        (
            "creative_writing",
            r"\b(?:essay|poem|poetry|haiku|song lyrics|joke)\b",
        ),
        (
            "trivia",
            r"\b(?:capital of|population of|who\s+(?:is|was)\s+the\s+president|"
            r"who\s+won|history of (?:the world|rome|china|america|europe)|"
            r"quantum physics|theory of relativity|periodic table)\b",
        ),
        (
            "weather_sports",
            r"\b(?:weather|forecast|temperature|sports?\s+score|football|basketball|"
            r"baseball|soccer|cricket)\b",
        ),
        (
            "food_travel",
            r"\b(?:recipe|cook|baking|restaurant recommendation|where to eat|"
            r"travel itinerary|book\s+(?:a|my)\s+(?:flight|hotel)|vacation|"
            r"holiday destination)\b",
        ),
        (
            "finance_trivia",
            r"\b(?:stock price|stock market|bitcoin|cryptocurrency|crypto price|"
            r"exchange rate|forex)\b",
        ),
        (
            "entertainment",
            r"\b(?:movie|film|netflix|celebrity|gossip|horoscope|astrology)\b",
        ),
        (
            "regulated_advice",
            r"\b(?:medical|diagnose|symptoms|legal advice|lawyer|lawsuit|"
            r"tax advice|investment advice)\b",
        ),
        (
            "homework",
            r"\b(?:homework|math problem|solve for x|derivative|integral|equation)\b",
        ),
        (
            "translation",
            r"\b(?:translate|translation)\s+(?:this|the following)\b",
        ),
        (
            "resume_writing",
            r"\b(?:write|draft|compose)\b.{0,40}\b(?:resume|cv|cover letter|email for me)\b",
        ),
    )
)


def _default_topic_similarity(message: str) -> float:
    """Use the knowledge-base retriever in chain.py without a circular import."""
    try:
        from .chain import get_topic_similarity

        return float(get_topic_similarity(message))
    except Exception as exc:  # Retrieval must not make the API unavailable.
        logger.warning("Topic similarity lookup failed: %s", exc)
        return 0.0


def _off_topic_match(normalized: str) -> str | None:
    for name, pattern in _OFF_TOPIC_PATTERNS:
        if pattern.search(normalized):
            return name
    return None


def check_topic_scope(
    message: str,
    similarity_fn: Callable[[str], float] | None = None,
) -> GuardrailDecision:
    """Keep the agent on product, pricing, lead-capture, and support topics."""
    normalized = normalize_text(message)
    if not normalized:
        return _blocked("input", "empty_message", "Please send a message so I can help.")

    if _GREETING_RE.fullmatch(normalized):
        return _allowed("input", "greeting")

    off_topic = _off_topic_match(normalized)
    if off_topic:
        return _blocked("input", "off_topic", OFF_TOPIC_BLOCKED, category=off_topic)

    if _IN_SCOPE_RE.search(normalized) or _FOLLOW_UP_RE.search(normalized):
        return _allowed("input", "in_scope_intent")

    threshold = _env_float(
        "GUARDRAIL_TOPIC_SIMILARITY_THRESHOLD",
        DEFAULT_TOPIC_SIMILARITY_THRESHOLD,
    )
    scorer = similarity_fn or _default_topic_similarity
    try:
        score = float(scorer(message))
    except Exception as exc:
        logger.warning("Topic similarity scorer failed: %s", exc)
        score = 0.0

    metadata = {"similarity": round(score, 4), "threshold": threshold}
    if score >= threshold:
        return _allowed("input", "in_scope_similarity", **metadata)
    return _blocked("input", "off_topic", OFF_TOPIC_BLOCKED, **metadata)


# ---------------------------------------------------------------------------
# Layer 1: payload, history, rate limit, injection, and scope checks
# ---------------------------------------------------------------------------


class SlidingWindowRateLimiter:
    """Thread-safe per-session sliding-window request and character limiter."""

    def __init__(self) -> None:
        self._events: dict[str, deque[tuple[float, int]]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(
        self,
        session_id: str,
        characters: int,
        *,
        max_requests: int,
        max_characters: int,
        window_seconds: int,
    ) -> tuple[bool, str | None, int]:
        now = time.monotonic()
        cutoff = now - max(1, window_seconds)
        key = session_id or "anonymous"
        with self._lock:
            events = self._events[key]
            while events and events[0][0] <= cutoff:
                events.popleft()
            request_count = len(events)
            character_count = sum(item[1] for item in events)
            if request_count >= max(1, max_requests):
                return False, "rate_limit", request_count
            if character_count + characters > max(1, max_characters):
                return False, "rate_limit_chars", character_count
            events.append((now, max(0, characters)))
            return True, None, request_count + 1

    def reset(self) -> None:
        with self._lock:
            self._events.clear()


_rate_limiter = SlidingWindowRateLimiter()


def reset_guardrail_state() -> None:
    """Clear in-memory rate-limit state (useful for tests and demos)."""
    _rate_limiter.reset()


def _history_stats(history: Iterable[Any] | None) -> tuple[int, int]:
    turns = list(history or [])
    total_chars = 0
    for turn in turns:
        if isinstance(turn, dict):
            content = turn.get("content", "")
        else:
            content = getattr(turn, "content", "")
        total_chars += len(str(content or ""))
    return len(turns), total_chars


def _looks_like_token_drain(message: str) -> bool:
    if len(message) < 500:
        return False
    words = re.findall(r"[A-Za-z0-9']+", message.lower())
    if len(words) < 100:
        return False
    unique_ratio = len(set(words)) / len(words)
    return unique_ratio < 0.12


def check_input(
    message: str,
    session_id: str,
    history: Iterable[Any] | None = None,
) -> GuardrailDecision:
    """Run every Layer 1 check before the main model is invoked."""
    text = message or ""
    max_chars = _env_int("GUARDRAIL_MAX_MESSAGE_CHARS", DEFAULT_MAX_MESSAGE_CHARS)
    if len(text) > max_chars:
        return _blocked(
            "input",
            "message_too_long",
            MESSAGE_TOO_LONG.format(limit=max_chars),
            characters=len(text),
            limit=max_chars,
        )
    if not text.strip():
        return _blocked("input", "empty_message", "Please send a message so I can help.")
    if _looks_like_token_drain(text):
        return _blocked(
            "input",
            "token_drain",
            TOKEN_DRAIN_BLOCKED,
            characters=len(text),
        )

    max_turns = _env_int("GUARDRAIL_MAX_HISTORY_TURNS", DEFAULT_MAX_HISTORY_TURNS)
    max_history_chars = _env_int("GUARDRAIL_MAX_HISTORY_CHARS", DEFAULT_MAX_HISTORY_CHARS)
    turn_count, history_chars = _history_stats(history)
    if turn_count > max_turns or history_chars > max_history_chars:
        return _blocked(
            "input",
            "history_limit",
            HISTORY_TOO_LONG,
            turns=turn_count,
            max_turns=max_turns,
            characters=history_chars,
            max_characters=max_history_chars,
        )

    injection = detect_prompt_injection(text)
    if injection:
        return _blocked("input", "prompt_injection", INJECTION_BLOCKED, category=injection)

    max_requests = _env_int("GUARDRAIL_RATE_LIMIT_REQUESTS", DEFAULT_RATE_LIMIT_REQUESTS)
    max_rate_chars = _env_int("GUARDRAIL_RATE_LIMIT_CHARS", DEFAULT_RATE_LIMIT_CHARS)
    window_seconds = _env_int(
        "GUARDRAIL_RATE_LIMIT_WINDOW_SECONDS",
        DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
    )
    allowed, rate_code, observed = _rate_limiter.check(
        session_id,
        len(text),
        max_requests=max_requests,
        max_characters=max_rate_chars,
        window_seconds=window_seconds,
    )
    if not allowed:
        return _blocked(
            "input",
            rate_code or "rate_limit",
            RATE_LIMITED,
            observed=observed,
            max_requests=max_requests,
            window_seconds=window_seconds,
        )

    return check_topic_scope(text)


# ---------------------------------------------------------------------------
# Layer 2: strict tool input validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    values: dict[str, Any] = field(default_factory=dict)
    code: str = ""
    message: str = ""
    field: str = ""


_EMAIL_RE = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$"
)
_PHONE_RE = re.compile(r"^[0-9+()\-.\s]+$")


def _clean_required(value: Any, field_name: str, max_chars: int) -> tuple[str | None, ValidationResult | None]:
    text = str(value or "").strip()
    if not text:
        return None, ValidationResult(
            ok=False,
            code=f"invalid_{field_name}",
            message=f"{field_name} is required and cannot be empty.",
            field=field_name,
        )
    if len(text) > max_chars:
        return None, ValidationResult(
            ok=False,
            code=f"invalid_{field_name}",
            message=f"{field_name} must be {max_chars} characters or fewer.",
            field=field_name,
        )
    return text, None


def validate_email(value: Any) -> tuple[str | None, ValidationResult | None]:
    text = str(value or "").strip()
    if not text:
        return None, ValidationResult(
            ok=False,
            code="invalid_email",
            message="email is required and cannot be empty.",
            field="email",
        )
    if len(text) > MAX_EMAIL_CHARS or not _EMAIL_RE.fullmatch(text):
        return None, ValidationResult(
            ok=False,
            code="invalid_email",
            message="email must be a valid address such as name@example.com.",
            field="email",
        )
    local_part, _, domain = text.rpartition("@")
    if len(local_part) > 64 or ".." in text or domain.startswith("-") or domain.endswith("-"):
        return None, ValidationResult(
            ok=False,
            code="invalid_email",
            message="email must be a valid address such as name@example.com.",
            field="email",
        )
    return text.lower(), None


def validate_phone(value: Any) -> tuple[str | None, ValidationResult | None]:
    text = str(value or "").strip()
    if not text:
        return None, None
    if len(text) > MAX_PHONE_CHARS or not _PHONE_RE.fullmatch(text):
        return None, ValidationResult(
            ok=False,
            code="invalid_phone",
            message="phone may contain only digits, spaces, parentheses, dots, hyphens, and a leading +.",
            field="phone",
        )
    if text.count("+") > 1 or ("+" in text and not text.startswith("+")):
        return None, ValidationResult(
            ok=False,
            code="invalid_phone",
            message="phone may contain only one leading + sign.",
            field="phone",
        )
    digits = re.sub(r"\D", "", text)
    if not 7 <= len(digits) <= 15:
        return None, ValidationResult(
            ok=False,
            code="invalid_phone",
            message="phone must contain between 7 and 15 digits.",
            field="phone",
        )
    return text, None


def _normalize_level(value: Any, field_name: str) -> tuple[str | None, ValidationResult | None]:
    text = str(value or "medium").strip().lower()
    if len(text) > MAX_PRIORITY_CHARS or text not in {"low", "medium", "high"}:
        return None, ValidationResult(
            ok=False,
            code=f"invalid_{field_name}",
            message=f"{field_name} must be one of: low, medium, high.",
            field=field_name,
        )
    return text, None


def validate_lead_input(
    *,
    name: Any,
    email: Any,
    company: Any = None,
    phone: Any = None,
    issue_summary: Any = None,
    urgency: Any = "medium",
) -> ValidationResult:
    """Validate every field before ``save_lead`` is permitted to run."""
    clean_name, error = _clean_required(name, "name", MAX_NAME_CHARS)
    if error:
        return error
    clean_email, error = validate_email(email)
    if error:
        return error
    clean_phone, error = validate_phone(phone)
    if error:
        return error
    clean_company = str(company or "").strip() or None
    if clean_company and len(clean_company) > MAX_COMPANY_CHARS:
        return ValidationResult(
            ok=False,
            code="invalid_company",
            message=f"company must be {MAX_COMPANY_CHARS} characters or fewer.",
            field="company",
        )
    clean_summary, error = _clean_required(issue_summary, "issue_summary", MAX_LEAD_SUMMARY_CHARS)
    if error:
        return error
    clean_urgency, error = _normalize_level(urgency, "urgency")
    if error:
        return error
    return ValidationResult(
        ok=True,
        values={
            "name": clean_name,
            "email": clean_email,
            "company": clean_company,
            "phone": clean_phone,
            "issue_summary": clean_summary,
            "urgency": clean_urgency,
        },
    )


def validate_ticket_input(
    *,
    user_name: Any,
    issue_description: Any,
    email: Any,
    priority: Any = "medium",
) -> ValidationResult:
    """Validate every field before ``save_ticket`` is permitted to run."""
    clean_name, error = _clean_required(user_name, "user_name", MAX_NAME_CHARS)
    if error:
        return error
    clean_summary, error = _clean_required(
        issue_description,
        "issue_description",
        MAX_TICKET_SUMMARY_CHARS,
    )
    if error:
        return error
    clean_email, error = validate_email(email)
    if error:
        return error
    clean_priority, error = _normalize_level(priority, "priority")
    if error:
        return error
    return ValidationResult(
        ok=True,
        values={
            "user_name": clean_name,
            "issue_description": clean_summary,
            "email": clean_email,
            "priority": clean_priority,
        },
    )


# ---------------------------------------------------------------------------
# Layer 3: output guardrails
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_MARKERS = (
    "you are dot, a concise business assistant",
    "never offer long filler text",
    "interactively collect: 1. full name",
    "call capture_lead first",
    "only after lead capture succeeds",
    "leads.db",
    "tickets.db",
    "system prompt",
)

_INTERNAL_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b[a-z]:\\[^\s\"']+"),
    re.compile(r"(?i)\bbackend[\\/](?:main|chain|db|guardrails)\.py\b"),
    re.compile(r"(?i)(?:/home/|/users/|/var/|/tmp/)[^\s\"']+"),
    re.compile(r"(?i)\bsite-packages\b"),
    re.compile(r"(?i)\btraceback\s+\(most recent call last\)"),
    re.compile(r'(?i)\bfile\s+"[^"]+",\s+line\s+\d+'),
)

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{8,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:ANTHROPIC_API_KEY|DB_ENCRYPTION_KEY|OPENAI_API_KEY)\b"),
    re.compile(r"\bapi[_-]?key\s*[:=]\s*\S{6,}", re.IGNORECASE),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{12,}\b"),
    re.compile(r"\bpassword\s*[:=]\s*\S{8,}", re.IGNORECASE),
)

_OUTPUT_DRIFT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE | re.DOTALL))
    for name, pattern in (
        (
            "code",
            r"```(?:python|javascript|typescript|java|sql|html|css|c\+\+|bash|powershell)",
        ),
        (
            "code",
            r"\bhere(?:'s| is) (?:the|a) (?:python|javascript|code|function|program)\b",
        ),
        (
            "creative",
            r"\b(?:once upon a time|here(?:'s| is) (?:a|your) (?:poem|essay|story|joke))\b",
        ),
        (
            "meta",
            r"\b(?:as an ai language model|i am a large language model)\b",
        ),
    )
)


def check_output(text: str, *, user_message: str | None = None) -> GuardrailDecision:
    """Inspect generated assistant text before any of it is sent to the client."""
    output = text or ""
    lowered = _strip_obfuscation(output).casefold()

    for marker in _SYSTEM_PROMPT_MARKERS:
        if marker in lowered:
            return _blocked(
                "output",
                "output_leak_system_prompt",
                OUTPUT_FALLBACK_MESSAGE,
                category="system_prompt_marker",
            )

    for pattern in _INTERNAL_PATH_PATTERNS:
        if pattern.search(output):
            return _blocked(
                "output",
                "output_leak_internal",
                OUTPUT_FALLBACK_MESSAGE,
                category="internal_path_or_stack_trace",
            )

    for pattern in _SECRET_PATTERNS:
        if pattern.search(output):
            return _blocked(
                "output",
                "output_leak_secret",
                OUTPUT_FALLBACK_MESSAGE,
                category="secret_pattern",
            )

    for name, pattern in _OUTPUT_DRIFT_PATTERNS:
        if pattern.search(output):
            return _blocked(
                "output",
                "output_off_topic",
                OUTPUT_FALLBACK_MESSAGE,
                category=name,
            )

    return _allowed("output", metadata={"characters": len(output)})


def safe_output(text: str, *, user_message: str | None = None) -> str:
    """Return the original text only when every output check passes."""
    decision = check_output(text, user_message=user_message)
    return text if decision.allowed else OUTPUT_FALLBACK_MESSAGE
