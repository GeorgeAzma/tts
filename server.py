import torchaudio as ta
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from chatterbox.tts import ChatterboxTTS
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from pathlib import Path
import torch
import asyncio
import shutil
import os
import sys
import io

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
    model: Optional[str] = ""
    input: str
    voice: Optional[str] = "elise"
    exaggeration: Optional[float] = 0.3
    cfg_weight: Optional[float] = 0.8
    temperature: Optional[float] = 0.3


@app.post("/v1/audio/speech")
async def tts(req: TTSRequest):
    try:
        global model, last_used
        if model is None:
            print("Loading model")
            model = ChatterboxTTS.from_pretrained(device="cuda")
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
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"Fatal CUDA error: {e}. Restarting...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        raise


@app.get("/v1/voices")
async def list_voices():
    """Return list of available voices"""
    if not voice_cache:
        load_voice_cache()
    return {"voices": list(voice_cache.keys())}


@app.post("/v1/voices/custom")
async def upload_custom_voice(voice: UploadFile = File(...)):
    """Upload a custom voice file"""

    # Validate file type
    if not voice.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file")

    voices_dir = Path("voices")
    voices_dir.mkdir(exist_ok=True)

    custom_voice_path = voices_dir / "custom.wav"

    try:
        with open(custom_voice_path, "wb") as buffer:
            shutil.copyfileobj(voice.file, buffer)

        # Reload voice cache to include the new custom voice
        load_voice_cache()

        return {
            "message": "Custom voice uploaded successfully",
            "filename": voice.filename,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save custom voice: {str(e)}"
        )


@app.get("/v1/voices/custom/preview")
async def preview_custom_voice():
    """Return the custom voice file for preview"""
    custom_voice_path = Path("voices") / "custom.wav"

    if not custom_voice_path.exists():
        raise HTTPException(status_code=404, detail="No custom voice found")

    return FileResponse(
        path=str(custom_voice_path), media_type="audio/wav", filename="custom.wav"
    )


@app.delete("/v1/voices/custom")
async def remove_custom_voice():
    """Remove the custom voice file"""
    custom_voice_path = Path("voices") / "custom.wav"

    if not custom_voice_path.exists():
        raise HTTPException(status_code=404, detail="No custom voice found")

    try:
        custom_voice_path.unlink()  # Delete the file

        # Reload voice cache to remove the custom voice
        load_voice_cache()

        return {"message": "Custom voice removed successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to remove custom voice: {str(e)}"
        )


@app.post("/v1/model/unload")
async def unload_model():
    global model
    if model is not None:
        del model
        model = None
        torch.cuda.empty_cache()
        return {"message": "Model unloaded successfully."}
    else:
        return {"message": "Model is already unloaded."}


from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=5000)
