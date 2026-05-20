"""
Gallery Guide — FastAPI Backend
Endpoints: /chat-stream, /image-search-stream, /sessions, /health
"""
# uvicorn main:app --reload --port 8001
import base64
import json
import os
import uuid
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from db import (
    create_session,
    delete_session,
    get_messages,
    get_sessions,
    save_message,
    update_session_title,
)
from models import ChatRequest
from prompt import museum_prompt, rewrite_prompt, vision_prompt
from rag import rag

load_dotenv()

OPENAI_KEY   = os.getenv("OPENAI_API_KEY", "")
OPENAI_URL   = "https://api.openai.com/v1/chat/completions"
CHAT_MODEL   = "gpt-4o-mini"
VISION_MODEL = "gpt-4o-mini"
MAX_TOKENS   = 500
CACHE: dict[str, str] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Gallery Guide backend starting…")
    yield
    print("Gallery Guide backend shutting down…")


app = FastAPI(title="Gallery Guide API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



# ── Helpers ───────────────────────────────────────────────────────────

def build_context(results: list[dict]) -> str:
    ctx = ""
    for r in results[:3]:
        text = (r.get("text") or "")[:400]
        ctx += f"\n— {r.get('title', '')} by {r.get('artist', '')} —\n{text}\n"
    return ctx.strip()


def get_sources(results: list[dict], top_artist: str = "") -> list[dict]:
    """
    First source = primary artwork (top result).
    Related sources = same artist only, to avoid irrelevant suggestions.
    """
    if not results:
        return []

    primary = {
        "title":     results[0].get("title", ""),
        "artist":    results[0].get("artist", ""),
        "year":      results[0].get("year", ""),
        "image_url": results[0].get("image_url", ""),
        "type":      results[0].get("type", "artwork"),
    }

    related = []
    artist = top_artist or results[0].get("artist", "")
    for r in results[1:]:
        if r.get("artist", "") == artist and r.get("image_url"):
            related.append({
                "title":     r.get("title", ""),
                "artist":    r.get("artist", ""),
                "year":      r.get("year", ""),
                "image_url": r.get("image_url", ""),
                "type":      r.get("type", "artwork"),
            })

    return [primary] + related


async def rewrite_query(question: str, db_history: list[dict]) -> tuple[str, bool]:
    # همیشه rewrite می‌کنیم تا زبان رو به انگلیسی تبدیل کنیم
    last_assistant = next(
        (m["content"][:300] for m in reversed(db_history) if m["role"] == "assistant"),
        "",
    )
    summary = f"Previous answer was about: {last_assistant}" if last_assistant else ""

    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.post(
                OPENAI_URL,
                headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
                json={
                    "model": CHAT_MODEL,
                    "messages": [{"role": "user", "content": rewrite_prompt(question, summary)}],
                    "max_tokens": 80,
                    "temperature": 0,
                },
            )
            result = json.loads(r.json()["choices"][0]["message"]["content"])
            return result.get("query", question), result.get("is_specific", True)
    except Exception:
        return question, True


def detect_has_artwork(question: str, results: list[dict], is_specific: bool) -> bool:
    if not results or not results[0].get("image_url"):
        return False
    top = results[0]
    title_words  = {w for w in top.get("title", "").lower().split() if len(w) > 3}
    artist_words = {w for w in top.get("artist", "").lower().split() if len(w) > 3}
    q = question.lower()
    return any(w in q for w in title_words) or any(w in q for w in artist_words) or is_specific


async def stream_openai(messages: list[dict]):
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream(
            "POST", OPENAI_URL,
            headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
            json={"model": CHAT_MODEL, "messages": messages, "stream": True, "max_tokens": MAX_TOKENS},
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        pass


# ── Routes ────────────────────────────────────────────────────────────

@app.post("/chat-stream")
async def chat_stream(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    # Load history from DB
    db_history = get_messages(session_id, limit=12)
    is_first_message = len(db_history) == 0

    # Cache check
    cache_key = f"{req.question.lower().strip()}::{req.language}"
    if is_first_message and cache_key in CACHE:
        cached = CACHE[cache_key]
        results = rag.search(req.question, limit=5)
        sources = get_sources(results)
        has_artwork = detect_has_artwork(req.question, results, True)

        async def cached_gen():
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'has_artwork': has_artwork})}\n\n"
            yield f"data: {json.dumps({'type': 'text', 'content': cached})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        create_session(session_id, req.language)  # ← اینجا
        save_message(session_id, "user", req.question)
        save_message(session_id, "assistant", cached, sources)
        return StreamingResponse(cached_gen(), media_type="text/event-stream")

    # Query rewrite
    search_query, is_specific = await rewrite_query(req.question, db_history)

    # RAG search
    results = rag.search(search_query, limit=5)
    context = build_context(results)
    sources = get_sources(results)
    has_artwork = detect_has_artwork(req.question, results, is_specific)

    # Build LLM messages
    system = museum_prompt(context, req.language)
    messages = [{"role": "system", "content": system}]
    for msg in db_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": req.question})

    # Session + user message — فقط موقع ارسال واقعی
    create_session(session_id, req.language)  # ← اینجا هم
    save_message(session_id, "user", req.question)

    full_answer = ""

    async def generate():
        nonlocal full_answer
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'has_artwork': has_artwork})}\n\n"

        async for delta in stream_openai(messages):
            full_answer += delta
            yield f"data: {json.dumps({'type': 'text', 'content': delta})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

        save_message(session_id, "assistant", full_answer, sources)

        if is_first_message:
            update_session_title(session_id, req.question[:60])
            CACHE[cache_key] = full_answer

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/image-search-stream")
async def image_search_stream(file: UploadFile = File(...)):
    image_data = await file.read()
    image_b64  = base64.b64encode(image_data).decode()

    # Vision: describe the image
    description = "An artwork image"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                OPENAI_URL,
                headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
                json={
                    "model": VISION_MODEL,
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{file.content_type};base64,{image_b64}"}},
                        {"type": "text", "text": vision_prompt()},
                    ]}],
                    "max_tokens": 200,
                },
            )
            description = r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Vision error: {e}")

    # RAG search
    results = rag.search_by_description(description, limit=5)
    context = build_context(results)
    sources = get_sources(results)

    system = museum_prompt(context, "en", image_description=description)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "What artwork is this? Tell me about it."},
    ]

    async def generate():
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'has_artwork': True})}\n\n"
        async for delta in stream_openai(messages):
            yield f"data: {json.dumps({'type': 'text', 'content': delta})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Session routes ────────────────────────────────────────────────────

@app.get("/sessions")
async def list_sessions():
    return get_sessions(limit=30)


@app.get("/sessions/{session_id}/messages")
async def list_messages(session_id: str):
    return get_messages(session_id, limit=50)


@app.post("/sessions")
async def new_session(language: str = "en"):
    sid = str(uuid.uuid4())
    create_session(sid, language)
    return {"id": sid}


@app.delete("/sessions/{session_id}")
async def remove_session(session_id: str):
    delete_session(session_id)
    return {"ok": True}


@app.get("/health")
async def health():
    return {"status": "ok", "collection": "gallery_guide"}