#!/usr/bin/env python
"""
Studio Process — "Enhance my take" backend for the nvh-the-noise VST.

Restoration / studio-cleanup using DeepFilterNet3 (keeps the performance,
removes noise/reverb-ish artifacts, makes a take studio-clean). Runs fully
isolated in this venv (Python 3.11 + torch CPU) so it can NOT touch the shared
portable ComfyUI env used by Krea2 / Qwen-TTS.

Usage:
    python enhance.py --input IN.wav --output OUT.wav [--atten-lim DB] [--keep-sr]

Prints a single JSON line to stdout on success:
    {"ok": true, "in_sr": 48000, "out_sr": 48000, "in_rms": .., "out_rms": .., "seconds": ..}
On failure prints {"ok": false, "error": "..."} and exits non-zero.
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Keep the model cache local to this folder so nothing leaks into the user's
# home cache and the download is reused across runs.
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_HERE, "model_cache"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--atten-lim",
        type=float,
        default=0.0,
        help="Max attenuation in dB (0 = full denoise). Higher = gentler / keeps more ambience.",
    )
    ap.add_argument(
        "--keep-sr",
        action="store_true",
        help="Resample the enhanced result back to the input sample rate (default: write 48 kHz).",
    )
    args = ap.parse_args()

    try:
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio
        from df.enhance import enhance, init_df, load_audio, save_audio

        if not os.path.isfile(args.input):
            raise FileNotFoundError(args.input)

        # Probe original sample rate (for optional resample-back).
        info = sf.info(args.input)
        in_sr = int(info.samplerate)

        model, df_state, _ = init_df(log_level="ERROR")
        model_sr = df_state.sr()  # 48000

        audio, _ = load_audio(args.input, sr=model_sr)  # (C, T) float32 @ 48k
        in_rms = float(torch.sqrt(torch.mean(audio.float() ** 2)).item())

        atten = args.atten_lim if args.atten_lim and args.atten_lim > 0 else None
        enhanced = enhance(model, df_state, audio, atten_lim_db=atten)

        out_sr = model_sr
        if args.keep_sr and in_sr != model_sr:
            enhanced = torchaudio.functional.resample(enhanced, model_sr, in_sr)
            out_sr = in_sr

        out_rms = float(torch.sqrt(torch.mean(enhanced.float() ** 2)).item())

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        save_audio(args.output, enhanced, out_sr)

        seconds = enhanced.shape[-1] / float(out_sr)
        print(json.dumps({
            "ok": True,
            "in_sr": in_sr,
            "out_sr": out_sr,
            "in_rms": round(in_rms, 6),
            "out_rms": round(out_rms, 6),
            "seconds": round(seconds, 3),
        }))
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
