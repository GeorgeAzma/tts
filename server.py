import torchaudio as ta
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from chatterbox.tts import ChatterboxTTS
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta, timezone
import io
import torch
import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path

voice_cache = {}


def load_voice_cache(voice_dir: str = "voices"):
    global voice_cache
    voice_cache.clear()

    if not os.path.exists(voice_dir):
        return

    for file_path in Path(voice_dir).iterdir():
        if file_path.is_file():
            voice_name = file_path.stem  # filename without extension
            voice_cache[voice_name] = str(file_path)


async def auto_unload():
    global model
    while True:
        await asyncio.sleep(60)
        if model and datetime.now(timezone.utc) - last_used > unload_delay:
            print("Unloading model")
            del model
            model = None
            torch.cuda.empty_cache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_voice_cache()
    asyncio.create_task(auto_unload())
    yield
    pass


app = FastAPI(lifespan=lifespan)
model = None
last_used = datetime.now(timezone.utc)
unload_delay = timedelta(minutes=5)


class TTSRequest(BaseModel):
    model: Optional[str] = "tts-1"
    input: str
    voice: Optional[str] = "elise"
    exaggeration: Optional[float] = 0.3
    cfg_weight: Optional[float] = 0.8
    temperature: Optional[float] = 0.3


@app.post("/v1/audio/speech")
async def tts(req: TTSRequest):
    global model, last_used
    if model is None:
        print("Loading model")
        model = ChatterboxTTS.from_local("chatterbox", "cuda")
    last_used = datetime.now(timezone.utc)

    if req.voice not in voice_cache:
        return {
            "error": f"Voice '{req.voice}' not found. Available voices: {list(voice_cache.keys())}"
        }

    wav = model.generate(
        text=req.input,
        audio_prompt_path=voice_cache[req.voice],
        exaggeration=req.exaggeration,
        cfg_weight=req.cfg_weight,
        temperature=req.temperature,
    )

    buffer = io.BytesIO()
    ta.save(buffer, wav, model.sr, format="wav")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="audio/wav")


@app.get("/v1/voices")
async def list_voices():
    """Return list of available voices"""
    if not voice_cache:
        load_voice_cache()
    return {"voices": list(voice_cache.keys())}


from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=5000)
