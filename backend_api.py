import base64
from datetime import datetime, timezone, timedelta
from typing import List, Optional
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import across_core as ac

app = FastAPI(title="ACROSS AI Backend", version="1.0.0")

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
    offline_only: bool = False
    image_base64: Optional[str] = None
    image_mime: Optional[str] = "image/jpeg"


class VoiceTokenRequest(BaseModel):
    history: List[ChatMessage] = []


@app.get("/health")
def health():
    return {
        "status": "ok",
        "app": "ACROSS",
        "corpus_entries": len(ac.LOCAL_CORPUS),
        "online": ac.check_internet_online(),
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

    is_online = (not req.offline_only) and ac.check_internet_online()

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
            img_res = ac.run_intelligent_image_pipeline(prompt, detected_script, allow_clarify=True)
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

    # 4. Online Hybrid Reasoning (Local Corpus + Math + Tavily + Groq)
    greetings = {"hi", "hello", "hey", "hii", "hola", "sup", "good morning", "good evening", "bye", "namaste", "pranam", "nomoskar"}
    search_triggers = ["winner", "jiteche", "won", "match", "score", "news", "current", "ajke", "ekhon", "latest", "2026", "who won", "today", "weather", "price"]
    needs_search = any(t in p_lower for t in search_triggers) and (p_lower not in greetings)

    local_hits = ac.retrieve_from_local_corpus(prompt, top_k=3)
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
        "You are ACROSS, an advanced sovereign hybrid AI assistant created exclusively by Shubham. "
        "If anyone asks who created, built, or owns you, always proudly state that you were created by Shubham. "
        "1. You natively understand and reply fluently in ALL 22 scheduled Indian languages and scripts as well as English, Banglish, and Hinglish. "
        "2. Always reply in the exact same language, script, and conversational register used by the user. "
        "3. Integrate any verified facts provided below from ACROSS's built-in Sovereign Knowledge Base with your own deep reasoning. "
        "4. LANGUAGE POLICY: you support ONLY English and Indian languages."
    )
    if local_ctx:
        system_content += f"\n\nACROSS Built-in Sovereign Knowledge Base Context:\n{local_ctx}"
    if search_ctx:
        system_content += f"\n\nLive Web Search Results:\n{search_ctx}"

    messages_payload = [{"role": "system", "content": system_content}]
    for m in req.history[-10:]:
        if m.role in ("user", "assistant"):
            messages_payload.append({"role": m.role, "content": m.content})
    messages_payload.append({"role": "user", "content": prompt})

    for key_idx in range(len(ac.KEYS_POOL)):
        api_key = ac.get_next_key(key_idx)
        for model_name in ac.get_available_models(api_key):
            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"model": model_name, "messages": messages_payload, "stream": False},
                    timeout=20,
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
    history_ctx = "\n".join(f"{m.role.title()}: {m.content[:250]}" for m in req.history[-6:])
    voice_sys_prompt = (
        "You are ACROSS, a warm, natural, sovereign voice AI assistant created exclusively by Shubham. "
        "STRICT IDENTITY RULE: Under NO circumstances should you ever say, utter, or mention 'Gemini', 'Google', 'Google AI', or any other company/model name. "
        "If anyone asks who you are, what AI/model you are, or who created or built you, always proudly say you are ACROSS, created by Shubham. "
        "Respond naturally in Bengali, Hindi, English, or any Indian language the user speaks. "
        "Keep spoken replies clear and conversational.\n\n"
        f"Recent conversation context:\n{history_ctx}"
    )
    try:
        return ac.create_ephemeral_live_token(api_key=gemini_key, system_instruction=voice_sys_prompt)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
