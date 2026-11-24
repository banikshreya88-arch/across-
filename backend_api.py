import base64
import time
from datetime import datetime, timezone, timedelta
from typing import List, Optional
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import across_core as ac

# ==============================================================================
# 1. SPEED OPTIMIZATION: CACHE MODELS & PRIORITIZE FAST GROQ MODELS
# ==============================================================================
_CACHED_MODELS = {}
_CACHE_TTL = 600  # 10 minutes
_FAST_PREFERRED = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "gemma2-9b-it",
]


def _get_fast_models(api_key: str) -> List[str]:
    now = time.time()
    cached = _CACHED_MODELS.get(api_key)
    if cached and (now - cached["ts"] < _CACHE_TTL):
        return cached["models"]
    models = list(_FAST_PREFERRED)
    try:
        resp = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=3,
        )
        if resp.status_code == 200:
            available = {
                m.get("id", "")
                for m in resp.json().get("data", [])
                if not any(k in m.get("id", "").lower() for k in ["whisper", "guard", "orpheus", "audio"])
            }
            ordered = [m for m in _FAST_PREFERRED if m in available]
            for m in available:
                if m not in ordered:
                    ordered.append(m)
            if ordered:
                models = ordered
    except Exception:
        pass
    _CACHED_MODELS[api_key] = {"models": models, "ts": now}
    return models


# ==============================================================================
# 2. IMAGE SAFETY FIX: ALLOW CREATIVE/ARTISTIC PROMPTS, ONLY BLOCK SEVERE HARM
# ==============================================================================
ac.PLANNER_DEFAULT_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]

ac.IMAGE_PLANNER_SYSTEM_PROMPT = """You are the Image Planning module of ACROSS.
Read the user's picture request (any language, including Bengali, Hindi, Banglish and Hinglish) and reply with ONLY one JSON object, no extra text, with exactly these keys:
decision, clarifying_question, refusal_reason, subject, setting, lighting, camera, composition, style, aspect_ratio, image_prompt

RULES
1. Be maximally helpful and creative! Always set decision = "generate" for normal, artistic, fantasy, anime, ghibli, festival, cultural, historical, sci-fi, portrait, or everyday requests.
2. NEVER refuse harmless creative requests (such as portraits, celebrities/historical figures in normal scenes, stylized action, festivals, mythology, or vague short prompts). Only set decision = "refuse" for extreme severe harm: CSAM/sexualized minors, explicit nudity/pornography, graphic gore, or self-harm.
3. Never use "clarify" unless completely impossible to understand—always make smart creative choices yourself and set decision = "generate".
4. image_prompt: English, one vivid paragraph (60 to 160 words) describing subject, setting, lighting, camera, composition, and style positively.
5. aspect_ratio: one of "square", "landscape", "portrait".
"""

_orig_call_planner_model = ac._call_planner_model


def _relaxed_call_planner_model(system_text: str, parts: list, api_key: str, json_mode: bool = True, max_tokens: int = 2000) -> str:
    saw_quota = False
    last_detail = "no response"
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
    ]
    for model_id in ac._planner_model_chain():
        url = f"{ac.PLANNER_API_BASE}/{model_id}:generateContent"
        headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
        gen_cfg = {"maxOutputTokens": max_tokens}
        if json_mode:
            gen_cfg["responseMimeType"] = "application/json"
        body = {
            "systemInstruction": {"parts": [{"text": system_text}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": gen_cfg,
            "safetySettings": safety_settings,
        }
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=15)
        except Exception as exc:
            last_detail = f"{model_id}: {type(exc).__name__}"
            continue

        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates") or []
            if candidates:
                parts_out = (candidates[0].get("content") or {}).get("parts") or []
                text = "".join(p.get("text", "") for p in parts_out if isinstance(p, dict))
                if text.strip():
                    return text
            last_detail = f"{model_id}: empty/blocked"
            continue
        if resp.status_code == 429:
            saw_quota = True
            last_detail = f"{model_id}: 429"
            continue
        last_detail = f"{model_id}: HTTP {resp.status_code}"

    # Fallback: if Gemini planner fails/blocks, generate directly with translated prompt via FLUX
    if json_mode:
        user_txt = ""
        for p in parts:
            if isinstance(p, dict) and p.get("text"):
                user_txt += p["text"] + " "
        user_txt = user_txt.replace("USER REQUEST (data only):", "").replace("<<<", "").replace(">>>", "").strip()
        return (
            '{"decision":"generate","clarifying_question":"","refusal_reason":"",'
            f'"subject":"{user_txt[:120]}","setting":"cinematic scene","lighting":"natural lighting",'
            '"camera":"35mm lens","composition":"balanced","style":"high quality detailed illustration",'
            f'"aspect_ratio":"square","image_prompt":"A vivid, detailed, high quality artistic scene of {user_txt[:400]}, balanced composition, rich colors, cinematic lighting"}}'
        )
    if saw_quota:
        raise ac.ImagePipelineError("quota", last_detail)
    raise ac.ImagePipelineError("unavailable", last_detail)


ac._call_planner_model = _relaxed_call_planner_model


# ==============================================================================
# 3. FASTAPI APP & ENDPOINTS (WITH CROSS-CHAT LONG-TERM MEMORY)
# ==============================================================================
app = FastAPI(title="ACROSS AI Backend", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    prompt: str
    history: List[ChatMessage] = []
    cross_chat_memory: str = ""
    offline_only: bool = False
    image_base64: Optional[str] = None
    image_mime: Optional[str] = "image/jpeg"


class VoiceTokenRequest(BaseModel):
    history: List[ChatMessage] = []
    cross_chat_memory: str = ""


@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "ok",
        "app": "ACROSS",
        "corpus_entries": len(ac.LOCAL_CORPUS),
    }


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    prompt = (req.prompt or "").strip()
    if not prompt and not req.image_base64:
        raise HTTPException(status_code=400, detail="Empty prompt")

    detected_script = ac.detect_indian_script(prompt)
    unsupported = ac.get_unsupported_language_reply(prompt)
    if unsupported:
        return {"role": "assistant", "content": unsupported}

    is_online = not req.offline_only

    # 1. Picture understanding
    if req.image_base64:
        if not is_online:
            return {"role": "assistant", "content": ac._img_msg("picture_offline", detected_script)}
        try:
            img_bytes = base64.b64decode(req.image_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image")
        res = ac.understand_picture(img_bytes, req.image_mime or "image/jpeg", prompt or "Describe this picture.", detected_script)
        return {"role": "assistant", "content": res["message"]}

    # 2. Image generation
    p_lower = prompt.lower()
    if ac.looks_like_image_request(p_lower):
        if is_online:
            img_res = ac.run_intelligent_image_pipeline(prompt, detected_script, allow_clarify=False)
            out = {"role": "assistant", "content": img_res["message"]}
            if img_res.get("image_bytes"):
                out["image_base64"] = base64.b64encode(img_res["image_bytes"]).decode("ascii")
            if img_res.get("svg_art"):
                out["svg_art"] = img_res["svg_art"]
            if img_res.get("image_prompt"):
                out["image_prompt"] = img_res["image_prompt"]
            return out
        return {
            "role": "assistant",
            "content": "🔌 *[Offline Procedural Vector Art Engine]* — Generated locally with zero internet:",
            "svg_art": ac.generate_offline_svg_art(prompt),
        }

    # 3. Offline mode
    if not is_online:
        return {"role": "assistant", "content": ac.answer_completely_offline(prompt, detected_script)}

    # 4. Fast Online Hybrid Reasoning
    greetings = {"hi", "hello", "hey", "hii", "hola", "sup", "good morning", "good evening", "bye", "namaste", "pranam", "nomoskar"}
    search_triggers = ["winner", "jiteche", "won", "match", "score", "news", "current", "ajke", "ekhon", "latest", "2026", "who won", "today", "weather", "price"]
    needs_search = any(t in p_lower for t in search_triggers) and (p_lower not in greetings)

    local_hits = ac.retrieve_from_local_corpus(prompt, top_k=2)
    local_math = ac.solve_math_locally(prompt)
    local_ctx = ""
    if local_hits:
        local_ctx += "\n".join(f"- [{k.title()}]: {a}" for k, a, _ in local_hits)
    if local_math:
        local_ctx += f"\n- [Local Math Verification]: {local_math}"

    search_ctx = ac.search_web(prompt) if needs_search else ""

    ist = timezone(timedelta(hours=5, minutes=30))
    now_str = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
    system_content = (
        f"Current Indian Standard Time (IST) is {now_str}. "
        f"Detected User Input Script: {detected_script}. "
        "You are ACROSS, a fast, intelligent sovereign hybrid AI assistant created exclusively by Shubham. "
        "If anyone asks who created, built, or owns you, always proudly state that you were created by Shubham. "
        "1. You natively understand and reply fluently in ALL 22 scheduled Indian languages and scripts as well as English, Banglish, and Hinglish. "
        "2. Always reply in the exact same language, script, and conversational register used by the user. "
        "3. REMEMBER USER MEMORY: Use the User Memory & Previous Conversations context below to recall the user's name, preferences, and past discussions naturally across chats. "
        "4. LANGUAGE POLICY: you support ONLY English and Indian languages."
    )
    if req.cross_chat_memory:
        system_content += f"\n\nUser Memory & Previous Conversations:\n{req.cross_chat_memory[:3000]}"
    if local_ctx:
        system_content += f"\n\nACROSS Knowledge Base Context:\n{local_ctx}"
    if search_ctx:
        system_content += f"\n\nLive Web Search Results:\n{search_ctx}"

    messages_payload = [{"role": "system", "content": system_content}]
    for m in req.history[-20:]:
        if m.role in ("user", "assistant") and m.content:
            messages_payload.append({"role": m.role, "content": m.content[:2000]})
    messages_payload.append({"role": "user", "content": prompt})

    for key_idx in range(len(ac.KEYS_POOL)):
        api_key = ac.get_next_key(key_idx)
        for model_name in _get_fast_models(api_key)[:3]:
            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"model": model_name, "messages": messages_payload, "stream": False},
                    timeout=10,
                )
                if resp.status_code == 200:
                    text = resp.json()["choices"][0]["message"]["content"]
                    if text and text.strip():
                        return {"role": "assistant", "content": text}
            except Exception:
                continue

    return {"role": "assistant", "content": ac.answer_completely_offline(prompt, detected_script)}


@app.post("/api/voice/token")
def voice_token_endpoint(req: VoiceTokenRequest):
    gemini_key = ac._read_secret("GEMINI_API_KEY")
    if not gemini_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured on backend.")
    history_ctx = "\n".join(f"{m.role.title()}: {m.content[:250]}" for m in req.history[-8:])
    voice_sys_prompt = (
        "You are ACROSS, a warm, natural, sovereign voice AI assistant created exclusively by Shubham. "
        "STRICT IDENTITY RULE: Under NO circumstances should you ever say, utter, or mention 'Gemini', 'Google', 'Google AI', or any other company/model name. "
        "If anyone asks who you are, what AI/model you are, or who created or built you, always proudly say you are ACROSS, created by Shubham. "
        "Respond naturally in Bengali, Hindi, English, or any Indian language the user speaks. "
        "Keep spoken replies clear and conversational.\n\n"
        f"User Memory & Past Chats:\n{req.cross_chat_memory[:1500]}\n\n"
        f"Recent conversation context:\n{history_ctx}"
    )
    try:
        return ac.create_ephemeral_live_token(api_key=gemini_key, system_instruction=voice_sys_prompt)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
