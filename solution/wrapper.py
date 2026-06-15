"""Observability + mitigation layer wrapping the opaque agent."""
from __future__ import annotations
import re
import time

try:
    from telemetry.logger import logger
    from telemetry.cost import cost_from_usage
    from telemetry.redact import redact
except Exception:
    logger = None
    def cost_from_usage(model, usage): return 0.0
    def redact(s): return (s, 0)

# Regex to detect prompt injection attempts in order notes
_NOTE_PATTERNS = [
    re.compile(r'(gia|price|total|tong)[^\w]*(la|is|=|:)\s*\d', re.IGNORECASE),
    re.compile(r'(ignore|bo qua|quen|forget|override).{0,30}(rule|prompt|instruction)', re.IGNORECASE),
    re.compile(r'(ghi\s*chu|note).{0,60}(gia|price|\d{4,})', re.IGNORECASE),
]

def _sanitize_question(question: str) -> str:
    """Strip injection attempts from order notes."""
    for pat in _NOTE_PATTERNS:
        question = pat.sub('[NOTE_REDACTED]', question)
    return question

def _validate_arithmetic(answer: str | None) -> bool:
    """Check answer contains a plausible VND total (integer)."""
    if not answer:
        return True  # refusal is valid
    m = re.search(r'[Tt]ong\s*cong\s*[:\-]?\s*([\d,\.]+)\s*VND', answer)
    if m:
        digits = re.sub(r'[,\.]', '', m.group(1))
        return digits.isdigit() and int(digits) > 0
    return True  # no total line = refusal, acceptable

def _has_pii(text: str) -> bool:
    """Quick PII check on answer."""
    _, count = redact(text)
    return count > 0

def _redact_answer(answer: str | None) -> str | None:
    """Redact PII from answer before returning."""
    if not answer:
        return answer
    redacted, _ = redact(answer)
    return redacted

def mitigate(call_next, question, config, context):
    session_id  = context.get("session_id", "")
    turn_index  = context.get("turn_index", 0)
    qid         = context.get("qid", "")
    cache       = context.get("cache", {})
    cache_lock  = context.get("cache_lock")

    # --- cache lookup (thread-safe) ---
    cache_key = f"{session_id}::{question}"
    if cache_lock:
        with cache_lock:
            cached = cache.get(cache_key)
    else:
        cached = cache.get(cache_key)
    if cached:
        _log("CACHE_HIT", qid, session_id, turn_index, cached, 0, 0.0, False)
        return cached

    # --- session drift reset ---
    if turn_index > 0 and turn_index % config.get("context_reset_every", 10) == 0:
        conf = dict(config)
        conf["session_id"] = f"{session_id}_reset_{turn_index}"
    else:
        conf = config

    # --- sanitize injection in question ---
    clean_question = _sanitize_question(question)

    # --- retry loop ---
    max_attempts = 3
    result = None
    for attempt in range(max_attempts):
        t0 = time.time()
        try:
            result = call_next(clean_question, conf)
        except Exception as exc:
            result = {"answer": None, "status": "wrapper_error", "steps": 0, "trace": [],
                      "meta": {"latency_ms": 0, "usage": {}, "tools_used": []}}
            _log("CALL_ERROR", qid, session_id, turn_index, result,
                 int((time.time()-t0)*1000), 0.0, False, error=str(exc))
            if attempt < max_attempts - 1:
                time.sleep(0.3 * (attempt + 1))
                continue
            return result

        wall_ms = int((time.time() - t0) * 1000)
        status  = result.get("status", "ok")
        meta    = result.get("meta", {})
        usage   = meta.get("usage", {})
        cost    = cost_from_usage(meta.get("model", ""), usage)
        answer  = result.get("answer")

        pii_found = _has_pii(answer or "")
        if pii_found:
            result["answer"] = _redact_answer(answer)

        _log("AGENT_CALL", qid, session_id, turn_index, result, wall_ms, cost, pii_found,
             attempt=attempt, tools=meta.get("tools_used", []),
             steps=result.get("steps", 0), status=status)

        # retry on transient errors only
        if status in ("ok", "loop", "max_steps", "no_action"):
            break
        if attempt < max_attempts - 1:
            time.sleep(0.3 * (attempt + 1))

    # --- store in cache ---
    if result and result.get("status") == "ok":
        if cache_lock:
            with cache_lock:
                cache[cache_key] = result
        else:
            cache[cache_key] = result

    return result


def _log(event, qid, session_id, turn_index, result, wall_ms, cost, pii_found,
         attempt=0, tools=None, steps=0, status="ok", error=None):
    if not logger:
        return
    meta  = result.get("meta", {}) if result else {}
    usage = meta.get("usage", {})
    data  = {
        "qid": qid,
        "session_id": session_id,
        "turn_index": turn_index,
        "status": status or result.get("status"),
        "wall_ms": wall_ms,
        "reported_latency_ms": meta.get("latency_ms"),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "cost_usd": cost,
        "tools_used": tools or [],
        "steps": steps,
        "pii_in_answer": pii_found,
        "attempt": attempt,
    }
    if error:
        data["error"] = error
    logger.log_event(event, data)
