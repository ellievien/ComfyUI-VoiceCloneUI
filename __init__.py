# ComfyUI-VoiceCloneUI
# Serves a standalone Windows 11-styled Voice Cloning page at /voiceclone
# that drives the FB_Qwen3TTSVoiceClone node via the existing ComfyUI API.
# Same-origin, so no CORS configuration is needed.

import os

from aiohttp import web

try:
    import folder_paths
except Exception:  # pragma: no cover
    folder_paths = None

from server import PromptServer

HERE = os.path.dirname(os.path.abspath(__file__))
QWEN_MODELS = [
    "Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen3-TTS-Tokenizer-12Hz",
]


def _input_dir():
    if folder_paths is not None:
        return folder_paths.get_input_directory()
    return os.path.join(os.getcwd(), "input")


def _models_root():
    if folder_paths is not None:
        return os.path.join(folder_paths.models_dir, "qwen-tts")
    return os.path.join(os.getcwd(), "models", "qwen-tts")


@PromptServer.instance.routes.get("/voiceclone")
async def voiceclone_index(request):
    return web.FileResponse(os.path.join(HERE, "index.html"))


@PromptServer.instance.routes.get("/voiceclone/status")
async def voiceclone_status(request):
    root = _models_root()
    models = {}
    for name in QWEN_MODELS:
        d = os.path.join(root, name)
        ok = os.path.isdir(d) and any(
            f.endswith(".safetensors") for f in (os.listdir(d) if os.path.isdir(d) else [])
        )
        # tokenizer has no safetensors check fallback: just dir exists + has files
        if name.endswith("Tokenizer-12Hz"):
            ok = os.path.isdir(d) and len(os.listdir(d)) > 1
        models[name] = ok
    return web.json_response({"models": models})


@PromptServer.instance.routes.post("/voiceclone/upload")
async def voiceclone_upload(request):
    reader = await request.multipart()
    field = await reader.next()
    if field is None:
        return web.json_response({"error": "no file"}, status=400)
    raw_name = os.path.basename(field.filename or "ref_audio.webm")
    # keep extension, sanitize stem
    stem, ext = os.path.splitext(raw_name)
    safe_stem = "".join(c for c in stem if c.isalnum() or c in ("-", "_")) or "ref_audio"
    if not ext:
        ext = ".webm"
    input_dir = _input_dir()
    os.makedirs(input_dir, exist_ok=True)
    filename = f"{safe_stem}{ext}"
    dest = os.path.join(input_dir, filename)
    i = 1
    while os.path.exists(dest):
        filename = f"{safe_stem}_{i}{ext}"
        dest = os.path.join(input_dir, filename)
        i += 1
    size = 0
    with open(dest, "wb") as f:
        while True:
            chunk = await field.read_chunk()
            if not chunk:
                break
            size += len(chunk)
            f.write(chunk)
    return web.json_response({"filename": filename, "size": size})


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

print("✅ VoiceCloneUI loaded — open http://127.0.0.1:8188/voiceclone")
