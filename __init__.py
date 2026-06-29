# ComfyUI-VoiceCloneUI
# Serves a standalone Windows 11-styled Voice Cloning page at /voiceclone
# that drives the FB_Qwen3TTSVoiceClone node via the existing ComfyUI API.
# Also exposes /voiceclone/transcribe (Whisper ASR) so the reference clip's
# transcript can be auto-filled instead of typed by hand.
# Same-origin, so no CORS configuration is needed.

import os
import json
import asyncio
import subprocess
import threading

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

# "Studio Process — Enhance my take" backend. DeepFilterNet can't run in this
# shared portable env (no cp313 libdf wheel + it would clash with the pinned
# torch/transformers), so it lives in a fully isolated Python 3.11 venv. We just
# shell out to it; nothing here imports it. See VocalStudioEnhance/enhance.py.
ENHANCE_VENV_PY = r"C:\Users\nvhstudio\VocalStudioEnhance\.venv\Scripts\python.exe"
ENHANCE_SCRIPT = r"C:\Users\nvhstudio\VocalStudioEnhance\enhance.py"
POLISH_SCRIPT  = r"C:\Users\nvhstudio\VocalStudioEnhance\polish.py"   # "Suno Polish" chain

# Small persistent config (currently just the stored "my voice" reference clip
# used by the plugin's Regenerate-in-my-cloned-voice mode).
CONFIG_PATH = os.path.join(HERE, "config.json")

# Whisper model used for auto-transcription. Prefer a local copy under
# models/whisper-small (avoids Windows symlink-privilege issues in the HF cache);
# fall back to the hub id if the local folder isn't present.
WHISPER_REPO = "openai/whisper-small"

_asr = None
_asr_lock = threading.Lock()


def _whisper_model_path():
    if folder_paths is not None:
        local = os.path.join(folder_paths.models_dir, "whisper-small")
        if os.path.isdir(local) and os.path.isfile(os.path.join(local, "config.json")):
            return local
    return WHISPER_REPO


def _input_dir():
    if folder_paths is not None:
        return folder_paths.get_input_directory()
    return os.path.join(os.getcwd(), "input")


def _output_dir():
    if folder_paths is not None:
        return folder_paths.get_output_directory()
    return os.path.join(os.getcwd(), "output")


def _models_root():
    if folder_paths is not None:
        return os.path.join(folder_paths.models_dir, "qwen-tts")
    return os.path.join(os.getcwd(), "models", "qwen-tts")


def _load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def _enhance_path(in_path, out_path, atten_lim=0.0, keep_sr=True):
    """Blocking: run the isolated DeepFilterNet venv on a file. Returns the
    parsed JSON metrics dict, or raises with the backend's error message."""
    if not os.path.isfile(ENHANCE_VENV_PY):
        raise RuntimeError(f"enhance venv missing: {ENHANCE_VENV_PY}")
    cmd = [ENHANCE_VENV_PY, ENHANCE_SCRIPT, "--input", in_path, "--output", out_path]
    if atten_lim and float(atten_lim) > 0:
        cmd += ["--atten-lim", str(float(atten_lim))]
    if keep_sr:
        cmd += ["--keep-sr"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    out = (proc.stdout or "").strip().splitlines()
    payload = None
    for line in reversed(out):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
                break
            except Exception:
                continue
    if payload is None:
        raise RuntimeError((proc.stderr or proc.stdout or "enhance failed").strip()[:500])
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error", "enhance failed"))
    return payload


def _polish_path(in_path, out_path, opts):
    """Blocking: run the isolated 'Suno Polish' chain (clean -> autotune ->
    production chain -> loud master) on a file. Returns the parsed metrics."""
    if not os.path.isfile(ENHANCE_VENV_PY):
        raise RuntimeError(f"polish venv missing: {ENHANCE_VENV_PY}")
    cmd = [ENHANCE_VENV_PY, POLISH_SCRIPT, "--input", in_path, "--output", out_path]
    cmd += ["--clean"]    if opts.get("clean", True)    else ["--no-clean"]
    cmd += ["--autotune"] if opts.get("autotune", True) else ["--no-autotune"]
    cmd += ["--key", str(opts.get("key", "C"))]
    cmd += ["--scale", str(opts.get("scale", "major"))]
    for flag, key in (("--tune-strength", "tune_strength"), ("--brightness", "brightness"),
                      ("--saturation", "saturation"), ("--width", "width"),
                      ("--space", "space"), ("--reverb", "reverb"), ("--loud", "loud")):
        if opts.get(key) is not None:
            cmd += [flag, str(float(opts[key]))]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    out = (proc.stdout or "").strip().splitlines()
    payload = None
    for line in reversed(out):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
                break
            except Exception:
                continue
    if payload is None:
        raise RuntimeError((proc.stderr or proc.stdout or "polish failed").strip()[:500])
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error", "polish failed"))
    return payload


def _get_asr():
    """Lazily build and cache the Whisper ASR pipeline (GPU if available)."""
    global _asr
    with _asr_lock:
        if _asr is None:
            import torch
            from transformers import pipeline
            use_cuda = torch.cuda.is_available()
            _asr = pipeline(
                "automatic-speech-recognition",
                model=_whisper_model_path(),
                device=0 if use_cuda else -1,
                torch_dtype=torch.float16 if use_cuda else torch.float32,
            )
        return _asr


def _transcribe_path(path, language=None):
    """Blocking: load audio with librosa (no ffmpeg needed) and run Whisper."""
    import librosa
    audio, _ = librosa.load(path, sr=16000, mono=True)
    asr = _get_asr()
    gen_kwargs = {"task": "transcribe"}
    if language and str(language).strip().lower() not in ("", "auto"):
        gen_kwargs["language"] = str(language).strip().lower()
    result = asr(
        {"array": audio, "sampling_rate": 16000},
        chunk_length_s=30,
        generate_kwargs=gen_kwargs,
    )
    return (result.get("text") or "").strip()


@PromptServer.instance.routes.get("/voiceclone")
async def voiceclone_index(request):
    return web.FileResponse(os.path.join(HERE, "index.html"))


@PromptServer.instance.routes.get("/voiceclone/status")
async def voiceclone_status(request):
    root = _models_root()
    models = {}
    for name in QWEN_MODELS:
        d = os.path.join(root, name)
        if name.endswith("Tokenizer-12Hz"):
            ok = os.path.isdir(d) and len(os.listdir(d)) > 1
        else:
            ok = os.path.isdir(d) and any(
                f.endswith(".safetensors") for f in (os.listdir(d) if os.path.isdir(d) else [])
            )
        models[name] = ok
    return web.json_response({"models": models})


@PromptServer.instance.routes.post("/voiceclone/upload")
async def voiceclone_upload(request):
    reader = await request.multipart()
    field = await reader.next()
    if field is None:
        return web.json_response({"error": "no file"}, status=400)
    raw_name = os.path.basename(field.filename or "ref_audio.webm")
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


@PromptServer.instance.routes.post("/voiceclone/transcribe")
async def voiceclone_transcribe(request):
    """Body: {"filename": "<file in input dir>", "language": "English"|"auto"}.
    Returns {"text": "..."}. Runs Whisper off the event loop."""
    try:
        data = await request.json()
    except Exception:
        data = {}
    filename = data.get("filename")
    language = data.get("language")
    if not filename:
        return web.json_response({"error": "no filename"}, status=400)
    path = os.path.join(_input_dir(), os.path.basename(filename))
    if not os.path.isfile(path):
        return web.json_response({"error": "file not found"}, status=404)
    try:
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, _transcribe_path, path, language)
        return web.json_response({"text": text})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.post("/voiceclone/enhance")
async def voiceclone_enhance(request):
    """Studio Process — "Enhance my take".
    Body: {"filename": "<file in input dir>", "atten_lim": 0, "keep_sr": true}.
    Runs the isolated DeepFilterNet venv off the event loop and writes a clean
    wav into the output dir. Returns {"filename","type":"output", metrics...}."""
    try:
        data = await request.json()
    except Exception:
        data = {}
    filename = data.get("filename")
    if not filename:
        return web.json_response({"error": "no filename"}, status=400)
    in_path = os.path.join(_input_dir(), os.path.basename(filename))
    if not os.path.isfile(in_path):
        return web.json_response({"error": "file not found"}, status=404)

    out_dir = _output_dir()
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(filename))[0]
    safe_stem = "".join(c for c in stem if c.isalnum() or c in ("-", "_")) or "take"
    out_name = f"enhanced_{safe_stem}.wav"
    out_path = os.path.join(out_dir, out_name)
    i = 1
    while os.path.exists(out_path):
        out_name = f"enhanced_{safe_stem}_{i}.wav"
        out_path = os.path.join(out_dir, out_name)
        i += 1

    atten_lim = data.get("atten_lim", 0.0) or 0.0
    keep_sr = bool(data.get("keep_sr", True))
    try:
        loop = asyncio.get_event_loop()
        metrics = await loop.run_in_executor(
            None, _enhance_path, in_path, out_path, atten_lim, keep_sr
        )
        metrics.update({"filename": out_name, "type": "output", "subfolder": ""})
        return web.json_response(metrics)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.post("/voiceclone/polish")
async def voiceclone_polish(request):
    """Studio Process — "Suno Polish".
    Body: {"filename", "clean"?, "autotune"?, "key"?, "scale"?, "tune_strength"?,
           "brightness"?, "saturation"?, "width"?, "space"?, "loud"?}.
    Runs the isolated DSP/autotune chain off the event loop and writes a
    produced wav into the output dir. Returns {"filename","type", metrics...}."""
    try:
        data = await request.json()
    except Exception:
        data = {}
    filename = data.get("filename")
    if not filename:
        return web.json_response({"error": "no filename"}, status=400)
    in_path = os.path.join(_input_dir(), os.path.basename(filename))
    if not os.path.isfile(in_path):
        return web.json_response({"error": "file not found"}, status=404)

    out_dir = _output_dir()
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(filename))[0]
    safe_stem = "".join(c for c in stem if c.isalnum() or c in ("-", "_")) or "take"
    out_name = f"suno_{safe_stem}.wav"
    out_path = os.path.join(out_dir, out_name)
    i = 1
    while os.path.exists(out_path):
        out_name = f"suno_{safe_stem}_{i}.wav"
        out_path = os.path.join(out_dir, out_name)
        i += 1

    try:
        loop = asyncio.get_event_loop()
        metrics = await loop.run_in_executor(None, _polish_path, in_path, out_path, data)
        metrics.update({"filename": out_name, "type": "output", "subfolder": ""})
        return web.json_response(metrics)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.get("/voiceclone/myvoice")
async def voiceclone_myvoice_get(request):
    """Returns the stored "my voice" reference clip (in the input dir) used by
    Regenerate mode: {"filename": <name|None>, "ref_text": <str>, "exists": bool}."""
    cfg = _load_config()
    fn = cfg.get("my_voice_filename")
    exists = bool(fn) and os.path.isfile(os.path.join(_input_dir(), os.path.basename(fn)))
    return web.json_response({
        "filename": fn,
        "ref_text": cfg.get("my_voice_ref_text", ""),
        "exists": exists,
    })


@PromptServer.instance.routes.post("/voiceclone/myvoice")
async def voiceclone_myvoice_set(request):
    """Body: {"filename": "<file already uploaded to input dir>", "ref_text": "..."}.
    Persists the stored "my voice" reference for Regenerate mode."""
    try:
        data = await request.json()
    except Exception:
        data = {}
    filename = data.get("filename")
    if not filename:
        return web.json_response({"error": "no filename"}, status=400)
    if not os.path.isfile(os.path.join(_input_dir(), os.path.basename(filename))):
        return web.json_response({"error": "file not found in input dir"}, status=404)
    cfg = _load_config()
    cfg["my_voice_filename"] = os.path.basename(filename)
    if "ref_text" in data:
        cfg["my_voice_ref_text"] = data.get("ref_text") or ""
    _save_config(cfg)
    return web.json_response({"ok": True, "filename": cfg["my_voice_filename"]})


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

print("✅ VoiceCloneUI loaded — http://127.0.0.1:8188/voiceclone (transcribe + Studio enhance/polish enabled)")
