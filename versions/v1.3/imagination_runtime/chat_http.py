"""
Shared HTTP routes for text/VLM chat: /generate, /api/chat, /health.

Used by ``colab_backend`` (FastAPI-only) and ``api_server`` (auth + API, no Gradio).
"""
from __future__ import annotations

import json
import os
from threading import Lock
from typing import Any, Dict, Iterator, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from imagination_runtime.chat_web_research import (
    final_text_from_stream,
    iterate_display_text_with_web,
    web_followups_enabled,
)

_GEN_LOCK = Lock()


def _web_followups_active() -> bool:
    """Web research loop is only enabled for text (non-VLM) stacks; VLM templates are unchanged."""
    if not web_followups_enabled():
        return False
    try:
        from imagination_v1_3 import RUNTIME

        if getattr(RUNTIME, "main_is_vlm", False):
            return False
    except Exception:
        return False
    return True


class ChatMessage(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., min_length=1)
    max_new_tokens: Optional[int] = None


class ChatApiRequest(BaseModel):
    """Payload from Next.js / v0 frontends (prompt + model + optional full history)."""

    prompt: str = ""
    currentModel: str = "imagination-1.3"
    messages: Optional[List[ChatMessage]] = None
    max_new_tokens: Optional[int] = None


def add_cors_from_env(app: FastAPI) -> None:
    _cors = (os.getenv("COLAB_BACKEND_CORS") or os.getenv("IMAGINATION_API_CORS") or "*").strip()
    if _cors == "*":
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[o.strip() for o in _cors.split(",") if o.strip()],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )


def _messages_dicts(items: List[ChatMessage]) -> List[Dict[str, str]]:
    return [{"role": m.role.strip(), "content": (m.content or "").strip()} for m in items]


def _stream_native(messages: List[Dict[str, str]], max_new_tokens: int) -> Iterator[str]:
    """Yield cumulative decoded text (same as ``generate_stream_chat`` / VLM stream)."""
    from imagination_v1_3 import RUNTIME, _effective_main_max_new_tokens, generate_stream_chat
    from imagination_runtime.vlm_infer import generate_stream_vlm

    if RUNTIME.main_tokenizer is None or RUNTIME.main_model is None:
        raise RuntimeError("Main model not loaded")

    cap = _effective_main_max_new_tokens()
    want = min(max(32, int(max_new_tokens)), cap)

    if RUNTIME.main_is_vlm and RUNTIME.main_processor is not None:
        yield from generate_stream_vlm(
            processor=RUNTIME.main_processor,
            tokenizer=RUNTIME.main_tokenizer,
            model=RUNTIME.main_model,
            messages=messages,
            max_new_tokens=want,
            lock=_GEN_LOCK,
            image=None,
        )
        return

    yield from generate_stream_chat(
        tokenizer=RUNTIME.main_tokenizer,
        model=RUNTIME.main_model,
        messages=messages,
        max_new_tokens=want,
        lock=_GEN_LOCK,
        extra_generate_kwargs=None,
    )


def _generate_native(messages: List[Dict[str, str]], max_new_tokens: int) -> str:
    out = ""
    for chunk in _stream_native(messages, max_new_tokens):
        out = chunk
    return out


def attach_generation_routes(app: FastAPI) -> None:
    """Register /generate, /api/chat, and a simple /health (in addition to /api/health from ASGI core)."""

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok", "model_name": "Imagination 1.3"}

    @app.post("/generate")
    def generate_completion(body: GenerateRequest) -> Dict[str, Any]:
        from imagination_v1_3 import _effective_main_max_new_tokens

        msgs = _messages_dicts(body.messages)
        if not any(m["content"] for m in msgs):
            raise HTTPException(status_code=400, detail="Empty message content")
        default_max = _effective_main_max_new_tokens()
        max_nt = body.max_new_tokens if body.max_new_tokens is not None else default_max
        try:
            text = _generate_native(msgs, max_nt)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        return {
            "response": text,
            "model": "Imagination 1.3",
            "max_new_tokens_effective": min(max_nt, default_max),
        }

    @app.post("/api/chat")
    def api_chat(body: ChatApiRequest) -> Dict[str, Any]:
        from imagination_v1_3 import _effective_main_max_new_tokens

        prompt = (body.prompt or "").strip()
        if body.messages:
            msgs = _messages_dicts(body.messages)
            if not any(m["content"] for m in msgs):
                raise HTTPException(status_code=400, detail="Empty message content")
        elif prompt:
            msgs = [{"role": "user", "content": prompt}]
        else:
            raise HTTPException(status_code=400, detail="Provide `prompt` and/or `messages`")
        default_max = _effective_main_max_new_tokens()
        max_nt = body.max_new_tokens if body.max_new_tokens is not None else default_max
        try:
            if _web_followups_active():
                text = final_text_from_stream(
                    iterate_display_text_with_web(
                        base_messages=msgs,
                        max_new_tokens=max_nt,
                        stream_native=_stream_native,
                    )
                )
            else:
                text = _generate_native(msgs, max_nt)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        return {
            "response": text,
            "model": body.currentModel or "Imagination 1.3",
            "max_new_tokens_effective": min(max_nt, default_max),
        }

    @app.post("/api/chat/stream")
    def api_chat_stream(body: ChatApiRequest) -> StreamingResponse:
        """NDJSON stream: one JSON object per line: ``{\"text\": \"...\"}`` (cumulative assistant text)."""
        from imagination_v1_3 import _effective_main_max_new_tokens

        prompt = (body.prompt or "").strip()
        if body.messages:
            msgs = _messages_dicts(body.messages)
            if not any(m["content"] for m in msgs):
                raise HTTPException(status_code=400, detail="Empty message content")
        elif prompt:
            msgs = [{"role": "user", "content": prompt}]
        else:
            raise HTTPException(status_code=400, detail="Provide `prompt` and/or `messages`")
        default_max = _effective_main_max_new_tokens()
        max_nt = body.max_new_tokens if body.max_new_tokens is not None else default_max

        def ndjson_chunks() -> Iterator[bytes]:
            try:
                if _web_followups_active():
                    stream_iter = iterate_display_text_with_web(
                        base_messages=msgs,
                        max_new_tokens=max_nt,
                        stream_native=_stream_native,
                    )
                else:
                    stream_iter = _stream_native(msgs, max_nt)

                for text in stream_iter:
                    line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
                    yield line.encode("utf-8")
            except Exception as e:
                err = json.dumps({"error": str(e)}, ensure_ascii=False) + "\n"
                yield err.encode("utf-8")

        return StreamingResponse(
            ndjson_chunks(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
