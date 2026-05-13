"""
modules/llm_client.py -- Native Gemini API Client

Stable Architecture LLM Routing:

  call_llm_local()       -> LM Studio ONLY (no Gemini fallback for these stages)
                            Use for: RAG insight extraction, Primary Diagnosis,
                                     3C3H Validation, MedAgentsBench benchmarks

  call_llm_gemini_only() -> Gemini ONLY (no LM Studio fallback)
                            Use for: Final Explanation, Final Report (optional)
                            These are the ONLY two stages that touch Gemini.

  call_llm()             -> Gemini first, LM Studio fallback (legacy helper,
                            retained for backward compatibility — prefer the
                            two explicit functions above for new call sites)
"""

from __future__ import annotations

import os
import time
import requests
from config import (
    GEMINI_API_KEY, GEMINI_MODEL,
    LM_STUDIO_URL, LM_STUDIO_TIMEOUT,
    FREE_TIER_RPM, FREE_TIER_RPD,
)

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

_RPM          = FREE_TIER_RPM
_GAP_SEC      = (60.0 / _RPM) + 1
_RPD          = FREE_TIER_RPD
_call_times:  list[float] = []
_calls_today: int         = 0
_today_date:  str         = ""
_last_backend: str        = "none"


def get_active_backend() -> str:
    return _last_backend


def gemini_budget_status() -> dict:
    _prune()
    return {
        "calls_in_last_minute": len(_call_times),
        "rpm_limit":            _RPM,
        "calls_today":          _calls_today,
        "rpd_limit":            _RPD,
        "next_call_in_sec":     _wait_needed(),
    }

openai_budget_status = gemini_budget_status


def _prune():
    global _call_times
    cutoff = time.time() - 60.0
    _call_times = [t for t in _call_times if t > cutoff]


def _wait_needed() -> float:
    _prune()
    if len(_call_times) >= _RPM:
        oldest = min(_call_times)
        return max(60.0 - (time.time() - oldest) + 1.5, 0.0)
    if _call_times:
        since_last = time.time() - max(_call_times)
        if since_last < _GAP_SEC:
            return _GAP_SEC - since_last
    return 0.0


def _record_call():
    global _call_times, _calls_today, _today_date
    import datetime
    today = datetime.date.today().isoformat()
    if today != _today_date:
        _today_date = today
        _calls_today = 0
    _calls_today += 1
    _call_times.append(time.time())


def _wait_for_slot():
    wait = _wait_needed()
    if wait > 0:
        b = gemini_budget_status()
        print(f"   [Gemini] Rate limit -- waiting {wait:.0f}s "
              f"({b['calls_in_last_minute']}/{_RPM} RPM, "
              f"{b['calls_today']}/{_RPD} today)")
        time.sleep(wait + 0.3)


def _get_key() -> str:
    key = GEMINI_API_KEY.strip()
    if not key:
        key = os.environ.get("GEMINI_API_KEY", "").strip()
    return key

_get_openai_key = _get_key


def _to_gemini_payload(messages: list[dict], temperature: float, max_tokens: int) -> dict:
    system_text = ""
    contents    = []
    for m in messages:
        role    = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            system_text = content
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": content}]})
        else:
            contents.append({"role": "user", "parts": [{"text": content}]})

    payload: dict = {
        "contents": contents,
        "generationConfig": {
            "temperature":     temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if system_text:
        payload["system_instruction"] = {"parts": [{"text": system_text}]}
    return payload


def _trim_messages(messages: list[dict], max_chars: int = 4000) -> list[dict]:
    return [
        {**m, "content": m["content"][:max_chars] + "\n[trimmed]"}
        if m["role"] == "user" and len(m["content"]) > max_chars
        else m
        for m in messages
    ]


def _call_gemini(messages: list[dict], temperature: float = 0.3, max_tokens: int = 500) -> str:
    key = _get_key()
    if not key:
        raise ValueError("No Gemini API key. Set GEMINI_API_KEY in config.py.")
    if _calls_today >= _RPD:
        raise RuntimeError(f"Daily limit reached ({_calls_today}/{_RPD}). Resets midnight UTC.")

    _wait_for_slot()
    url     = f"{_GEMINI_BASE}/{GEMINI_MODEL}:generateContent?key={key}"
    payload = _to_gemini_payload(messages, temperature, max_tokens)

    for attempt in range(3):
        _record_call()
        resp = requests.post(url, json=payload, timeout=60)

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", max(int(_GAP_SEC), 30)))
            print(f"   [Gemini] 429 -- waiting {retry_after}s (attempt {attempt+1}/3)...")
            time.sleep(retry_after)
            continue
        if resp.status_code == 400:
            detail = ""
            try:
                detail = resp.json().get("error", {}).get("message", "")
            except Exception:
                pass
            raise ValueError(f"Gemini 400 Bad Request: {detail}")
        if resp.status_code == 403:
            raise ValueError("Gemini 403 -- API key invalid or quota exceeded.")

        resp.raise_for_status()
        data = resp.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError):
            finish = data.get("candidates", [{}])[0].get("finishReason", "UNKNOWN")
            raise ValueError(f"Gemini returned no text (finishReason={finish})")
        if not text:
            raise ValueError("Gemini returned empty text.")
        return text

    raise RuntimeError("Gemini 429 persisted after 3 retries.")


def _get_lmstudio_model() -> str:
    try:
        resp = requests.get("http://localhost:1234/v1/models", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if models:
            return models[0].get("id", "")
    except Exception:
        pass
    return ""


def _merge_system_into_user(messages: list[dict]) -> list[dict]:
    """
    Merge system prompt into the first user message.
    Required for models that don't support the 'system' role (e.g. BioMistral, some Mistral variants).
    Format: [INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]
    """
    system_text = ""
    other = []
    for m in messages:
        if m.get("role") == "system":
            system_text = m["content"]
        else:
            other.append(m)

    if not system_text or not other:
        return messages

    merged = []
    first_user_done = False
    for m in other:
        if m.get("role") == "user" and not first_user_done:
            merged_content = f"<<SYS>>\n{system_text}\n<</SYS>>\n\n{m['content']}"
            merged.append({"role": "user", "content": merged_content})
            first_user_done = True
        else:
            merged.append(m)
    return merged


def _call_lmstudio(messages: list[dict], temperature: float = 0.3, max_tokens: int = 1024) -> str:
    loaded  = _get_lmstudio_model()

    def _attempt(msgs, include_model: bool) -> requests.Response:
        payload = {
            "messages":    msgs,
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "stream":      False,
        }
        if include_model and loaded:
            payload["model"] = loaded
        return requests.post(LM_STUDIO_URL, json=payload, timeout=LM_STUDIO_TIMEOUT)

    # Attempt 1: original messages with model field
    resp = _attempt(messages, include_model=True)

    if resp.status_code == 400:
        # Attempt 2: drop model field (some LM Studio versions reject explicit model names)
        resp = _attempt(messages, include_model=False)

    if resp.status_code == 400:
        # Attempt 3: merge system prompt into user turn (BioMistral / Mistral models)
        merged = _merge_system_into_user(messages)
        resp = _attempt(merged, include_model=False)

    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()
    if not content:
        raise ValueError("LM Studio returned empty content.")
    return content


def _reason(e: Exception) -> str:
    err = str(e)
    if "429" in err or "rate limit" in err.lower():     return "rate limited"
    if "daily limit" in err.lower():                    return "daily limit reached"
    if "403" in err or "invalid" in err.lower():        return "invalid API key -- check GEMINI_API_KEY in config.py"
    if "400" in err:                                    return f"bad request: {err[:100]}"
    if "Connection" in err or "refused" in err.lower(): return "not reachable"
    return err[:120]


def call_llm(messages: list[dict], temperature: float = 0.3, max_tokens: int = 1024) -> str:
    """
    Gemini first, LM Studio fallback.
    Use for: clinical explanation (short, safe text).
    """
    global _last_backend
    max_tokens = min(max_tokens, 4096)  # FIX 2: raised from 2048 to prevent truncation
    last_error = None

    if _get_key():
        try:
            result = _call_gemini(messages, temperature=temperature, max_tokens=max_tokens)
            _last_backend = "gemini"
            print(f"   [Gemini] OK {len(result)} chars")
            return result
        except Exception as e:
            print(f"   [Gemini] {_reason(e)}")
            last_error = e

    try:
        result = _call_lmstudio(messages, temperature=temperature, max_tokens=1024)
        _last_backend = "lmstudio"
        print(f"   [LM Studio] OK {len(result)} chars")
        return result
    except Exception as e:
        print(f"   [LM Studio] {_reason(e)}")
        last_error = e

    _last_backend = "none"
    return f"WARNING: All LLM backends failed. Last error: {last_error}"


def call_llm_local(messages: list[dict], temperature: float = 0.3, max_tokens: int = 1024) -> str:
    """
    LM Studio ONLY — no Gemini fallback.
    Use for: RAG insight extraction, Primary Diagnosis, 3C3H Validation, benchmarks.

    Stable Architecture: these pipeline stages must never reach Gemini so that
    Gemini quota is reserved exclusively for the Final Explanation and Final Report.
    If LM Studio is unavailable, a WARNING string is returned so the caller can
    handle degradation gracefully without silently consuming Gemini budget.
    """
    global _last_backend

    try:
        result = _call_lmstudio(messages, temperature=temperature, max_tokens=max_tokens)
        _last_backend = "lmstudio"
        print(f"   [LM Studio] OK {len(result)} chars")
        return result
    except Exception as e:
        reason = _reason(e)
        print(f"   [LM Studio] {reason} -- LM Studio unavailable (Gemini reserved for explanation/report)")

    _last_backend = "none"
    return "WARNING: LM Studio unavailable. Start LM Studio and load a model, then re-run."


def call_llm_gemini_only(messages: list[dict], temperature: float = 0.3, max_tokens: int = 1024) -> str:
    """
    Gemini ONLY — no LM Studio fallback.
    Use for: Final Explanation, Final Report (the ONLY two Gemini call sites).

    Stable Architecture: Gemini is reserved exclusively for patient-facing output.
    All upstream pipeline stages use call_llm_local() so Gemini quota is
    preserved for these two calls.
    """
    global _last_backend
    max_tokens = min(max_tokens, 4096)

    if _get_key():
        try:
            result = _call_gemini(messages, temperature=temperature, max_tokens=max_tokens)
            _last_backend = "gemini"
            print(f"   [Gemini] OK {len(result)} chars")
            return result
        except Exception as e:
            print(f"   [Gemini] {_reason(e)}")

    _last_backend = "none"
    return "WARNING: Gemini unavailable. Set GEMINI_API_KEY in config.py or check quota."


def build_messages(system_prompt: str, user_content: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]
