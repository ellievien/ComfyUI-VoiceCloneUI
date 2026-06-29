# studio_backend

The DSP/ML scripts behind the `/voiceclone/enhance` and `/voiceclone/polish` routes.
The ComfyUI node **shells out** to these (via `subprocess`) so none of this heavy ML is
imported into ComfyUI's own process — that keeps ComfyUI's pinned `torch` / `transformers`
untouched.

## Why a separate venv

DeepFilterNet's native `libdf` wheel and the pitch/DSP stack pin specific, older library
versions that conflict with a modern ComfyUI portable env. So these run in a **dedicated,
isolated Python 3.11 virtual environment**, completely separate from ComfyUI.

```bash
# create the isolated env (example with uv)
uv venv --python 3.11 .venv
# CPU torch is plenty — DeepFilterNet + the chain run faster than real time on CPU
uv pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
uv pip install deepfilternet DeepFilterLib soundfile        # enhance.py
uv pip install pedalboard librosa psola pyloudnorm           # polish.py / analyze.py
```

In the node (`__init__.py`), point these constants at your install:

```python
ENHANCE_VENV_PY = r"...\.venv\Scripts\python.exe"   # the isolated interpreter
ENHANCE_SCRIPT  = r"...\enhance.py"
POLISH_SCRIPT   = r"...\polish.py"
```

## Scripts

| Script | What it does | CLI |
|--------|--------------|-----|
| `enhance.py` | DeepFilterNet restoration (denoise, keep the performance) | `--input --output [--atten-lim DB] [--keep-sr]` |
| `polish.py`  | "Suno Polish": clean → autotune → production chain → loud master | `--input --output [--clean/--no-clean] [--autotune/--no-autotune] --key C --scale major\|minor\|chromatic --tune-strength 0.6 --brightness 1.0 --saturation 1.0 --width 1.0 --space 1.0 --reverb 0.30 --loud 1.0` |
| `analyze.py` | Profile a vocal (LUFS, crest, noise floor, octave-band balance, sibilance/mud, f0/key) to drive tuning | `--input` |

Each prints a single JSON line to stdout (`{"ok": true, ...}`), which the node parses.

## polish.py chain

`HPF 85 → gate → de-ess → leveler comp → corrective EQ (cut 200/300/400 Hz) →
restorative EQ (boost 3.3k/6k/11k) → saturation → glue comp → chorus(width) →
slap delay → reverb → LUFS-normalised loud master (-10 LUFS, hard ceiling -1 dBFS)`

The corrective cuts and restorative boosts default to a curve tuned for a dark, low-mid-heavy
close-mic vocal; `--brightness` scales the boosts and `--reverb` (0..1) drives the reverb knob.
