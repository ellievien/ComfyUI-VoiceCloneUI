#!/usr/bin/env python
"""
Studio Process — "Suno Polish" backend for the nvh-suno-polish VST.

Turns a raw recorded vocal into a produced, Suno-style vocal:
    clean (DeepFilterNet) -> autotune (snap to scale) -> production chain
    (HPF, gentle de-ess, leveling comp, corrective + restorative EQ, saturation,
     glue comp, width, slap delay, reverb) -> LUFS-normalized loud master.

The EQ + dynamics defaults are TUNED to the user's mic/voice baseline
(measured from "Track 10.wav": dark, low-mid-heavy at 120-300 Hz, recessed
presence/air, ~20 dB crest, -19.9 LUFS, not sibilant). See analyze.py.

Runs fully isolated in this venv (Python 3.11) so it can NOT touch the shared
portable ComfyUI env used by Krea2 / Qwen-TTS.

Usage:
    python polish.py --input IN.wav --output OUT.wav
        [--clean/--no-clean] [--autotune/--no-autotune]
        [--key C] [--scale major|minor|chromatic] [--tune-strength 0.6]
        [--brightness 1.0] [--saturation 1.0] [--width 1.0] [--space 1.0]
        [--reverb 0.30] [--loud 1.0]

Prints one JSON line on success (in/out peak, in/out LUFS, steps, seconds).
On failure: {"ok": false, "error": "..."} and exit non-zero.
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_HERE, "model_cache"))

WORK_SR = 44100
TARGET_LUFS = -10.0         # produced/"Suno" lead-vocal loudness target (clean+loud balance)
SCALES = {
    "major":     [0, 2, 4, 5, 7, 9, 11],
    "minor":     [0, 2, 3, 5, 7, 8, 10],
    "chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}
NOTE_PC = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5,
           "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11}


def _bool_flag(parser, name, default, help_):
    dest = name.replace("-", "_")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--" + name, dest=dest, action="store_true", help=help_)
    g.add_argument("--no-" + name, dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def _peak(x):
    import numpy as np
    return float(np.max(np.abs(x)) + 1e-12)


def _lufs(x, sr):
    """Integrated loudness (ITU BS.1770) via pyloudnorm; RMS-dB fallback."""
    import numpy as np
    try:
        import pyloudnorm as pyln
        return float(pyln.Meter(sr).integrated_loudness(np.ascontiguousarray(x, dtype=np.float64)))
    except Exception:
        return float(20.0 * np.log10(np.sqrt(np.mean(x ** 2) + 1e-12) + 1e-9))


def autotune(audio, sr, key, scale, strength):
    import numpy as np
    import librosa
    import psola

    fmin = librosa.note_to_hz("C2")
    fmax = librosa.note_to_hz("C6")
    frame_length = 2048
    hop_length = frame_length // 4

    f0, voiced_flag, _ = librosa.pyin(audio, sr=sr, fmin=fmin, fmax=fmax,
                                      frame_length=frame_length, hop_length=hop_length)
    if f0 is None or not np.any(np.isfinite(f0)):
        return audio

    key_pc = NOTE_PC.get(key.strip().upper(), 0)
    degrees = SCALES.get(scale.lower(), SCALES["major"])
    allowed = sorted({(key_pc + d) % 12 for d in degrees})

    corrected = np.copy(f0)
    for i, f in enumerate(f0):
        if not np.isfinite(f) or f <= 0:
            continue
        midi = librosa.hz_to_midi(f)
        base = int(np.round(midi))
        snapped = base
        for off in [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6]:
            if (base + off) % 12 in allowed:
                snapped = base + off
                break
        target_midi = (1.0 - strength) * midi + strength * snapped
        corrected[i] = librosa.midi_to_hz(target_midi)

    out = psola.vocode(audio, sample_rate=int(sr), target_pitch=corrected, fmin=fmin, fmax=fmax)
    return out.astype(np.float32)


def build_board(brightness, saturation, width, space, reverb, deess=1.0):
    """The tuned Suno-polish chain. Corrective EQ (cuts) is fixed to the mic
    profile; restorative EQ (boosts) scales with `brightness`. Reverb amount is
    its own control (the VST reverb knob), separate from `space` (slap delay)."""
    from pedalboard import (Pedalboard, HighpassFilter, LowShelfFilter, PeakFilter,
                            HighShelfFilter, Compressor, Distortion, Chorus, Delay, Reverb, NoiseGate)
    rv = float(max(0.0, min(1.0, reverb)))
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=85.0),                                  # rumble < C2
        NoiseGate(threshold_db=-48.0, ratio=2.0, attack_ms=2.0, release_ms=150.0), # tame residual floor in gaps
        PeakFilter(cutoff_frequency_hz=7200.0, gain_db=-2.0 * deess, q=2.5),       # gentle de-ess (not sibilant)
        Compressor(threshold_db=-21.0, ratio=3.0, attack_ms=12.0, release_ms=110.0),  # leveler: tame ~20 dB crest
        # --- corrective EQ (fixed; tuned to the mic) ---
        LowShelfFilter(cutoff_frequency_hz=200.0, gain_db=-3.5, q=0.9),            # tame 120-300 Hz proximity boom
        PeakFilter(cutoff_frequency_hz=300.0, gain_db=-3.0, q=1.2),                # de-mud
        PeakFilter(cutoff_frequency_hz=400.0, gain_db=-2.5, q=1.4),                # clear boxiness
        # --- restorative EQ (scales with brightness) ---
        PeakFilter(cutoff_frequency_hz=3300.0, gain_db=4.0 * brightness, q=1.0),   # presence/forwardness
        PeakFilter(cutoff_frequency_hz=6000.0, gain_db=1.5 * brightness, q=1.1),   # brilliance bridge
        HighShelfFilter(cutoff_frequency_hz=11000.0, gain_db=3.0 * brightness, q=0.7),  # air/sheen
        Distortion(drive_db=2.0 * saturation),                                     # harmonic warmth
        Compressor(threshold_db=-14.0, ratio=2.0, attack_ms=20.0, release_ms=150.0),   # glue
        Chorus(rate_hz=0.6, depth=0.12 * width, mix=0.18 * width),                 # width / doubling
        Delay(delay_seconds=0.11, feedback=0.12, mix=0.10 * space),                # slap
        Reverb(room_size=0.45, damping=0.6, wet_level=0.38 * rv, dry_level=1.0, width=1.0),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    _bool_flag(ap, "clean", True, "DeepFilterNet denoise before the chain")
    _bool_flag(ap, "autotune", True, "snap pitch toward the chosen scale")
    ap.add_argument("--key", default="C")
    ap.add_argument("--scale", default="major", choices=["major", "minor", "chromatic"])
    ap.add_argument("--tune-strength", type=float, default=0.6)
    ap.add_argument("--brightness", type=float, default=1.0)
    ap.add_argument("--saturation", type=float, default=1.0)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--space", type=float, default=1.0)
    ap.add_argument("--reverb", type=float, default=0.30, help="reverb amount 0..1 (the Reverb knob)")
    ap.add_argument("--loud", type=float, default=1.0, help="loudness scale; 1.0 -> -9.5 LUFS, higher = louder")
    _bool_flag(ap, "keep-sr", True, "write at the input sample rate (else 44.1k)")
    args = ap.parse_args()

    try:
        import numpy as np
        import soundfile as sf
        import librosa
        from pedalboard import Pedalboard, Limiter, Gain

        if not os.path.isfile(args.input):
            raise FileNotFoundError(args.input)

        steps = []
        audio, in_sr = sf.read(args.input, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        in_peak = _peak(audio)
        in_lufs = _lufs(audio, in_sr)

        # 1) clean (DeepFilterNet @ 48k) -> work SR
        if args.clean:
            import torch
            from df.enhance import enhance, init_df
            model, df_state, _ = init_df(log_level="ERROR")
            a48 = librosa.resample(audio, orig_sr=in_sr, target_sr=48000) if in_sr != 48000 else audio
            t = torch.from_numpy(np.ascontiguousarray(a48)).unsqueeze(0).float()
            enh = enhance(model, df_state, t).squeeze(0).cpu().numpy()
            work = librosa.resample(enh, orig_sr=48000, target_sr=WORK_SR)
            steps.append("clean")
        else:
            work = librosa.resample(audio, orig_sr=in_sr, target_sr=WORK_SR) if in_sr != WORK_SR else audio
        work = np.ascontiguousarray(work, dtype=np.float32)

        # normalise level so the chain's thresholds behave predictably (-3 dBFS peak)
        p = _peak(work)
        if p > 0:
            work = work * (10 ** (-3.0 / 20.0) / p)

        # 2) autotune
        if args.autotune:
            try:
                work = np.ascontiguousarray(
                    autotune(work, WORK_SR, args.key, args.scale, float(np.clip(args.tune_strength, 0.0, 1.0))),
                    dtype=np.float32)
                steps.append("autotune")
            except Exception as e:
                sys.stderr.write(f"autotune skipped: {e}\n")

        # 3) production chain
        board = build_board(args.brightness, args.saturation, args.width, args.space, args.reverb)
        work = board(work, WORK_SR)
        steps.append("chain")

        # 4) loud master: converge on the loudness target by iterating makeup
        #    gain into a -1 dBTP limiter (limiting density shifts LUFS, so a
        #    single normalize undershoots — 2-4 passes lock onto target).
        target = TARGET_LUFS + (float(np.clip(args.loud, 0.2, 2.0)) - 1.0) * 4.0
        pre = np.ascontiguousarray(work, dtype=np.float32)
        g = 0.0
        work = pre
        for _ in range(5):
            test = Pedalboard([Gain(gain_db=g), Limiter(threshold_db=-1.0, release_ms=100.0)])(pre, WORK_SR)
            m = _lufs(test, WORK_SR)
            work = test
            if not np.isfinite(m) or abs(m - target) <= 0.4:
                break
            g = float(np.clip(g + (target - m), -6.0, 24.0))
        steps.append("master")

        out_sr = in_sr if args.keep_sr else WORK_SR
        if out_sr != WORK_SR:
            work = librosa.resample(np.ascontiguousarray(work), orig_sr=WORK_SR, target_sr=out_sr)

        # final hard ceiling at -1 dBFS: CLIP stray overs (preserves loudness)
        # rather than scaling the whole signal down (which would undo the master).
        ceiling = 10 ** (-1.0 / 20.0)
        work = np.clip(work, -ceiling, ceiling)

        out_peak = _peak(work)
        out_lufs = _lufs(work, out_sr)

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        sf.write(args.output, work.astype(np.float32), out_sr)

        print(json.dumps({
            "ok": True,
            "in_sr": int(in_sr), "out_sr": int(out_sr),
            "in_peak": round(in_peak, 4), "out_peak": round(out_peak, 4),
            "in_lufs": round(in_lufs, 2), "out_lufs": round(out_lufs, 2),
            "reverb": round(float(args.reverb), 3),
            "steps": steps,
            "seconds": round(len(work) / float(out_sr), 3),
        }))
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
