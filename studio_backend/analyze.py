#!/usr/bin/env python
"""Profile a raw vocal take to drive Suno-polish tuning. Prints one JSON line."""
import argparse, json, sys, warnings
warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    a = ap.parse_args()
    try:
        import numpy as np, soundfile as sf, librosa, pyloudnorm as pyln

        x, sr = sf.read(a.input, dtype="float32")
        ch = 1 if x.ndim == 1 else x.shape[1]
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = np.ascontiguousarray(x, dtype=np.float32)
        n = len(x)
        dur = n / sr

        def dbfs(v):
            return float(20 * np.log10(max(v, 1e-9)))

        peak = float(np.max(np.abs(x)))
        rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
        dc = float(np.mean(x))
        clip = int(np.sum(np.abs(x) >= 0.99))

        # integrated loudness (ITU BS.1770)
        try:
            meter = pyln.Meter(sr)
            lufs = float(meter.integrated_loudness(x))
        except Exception:
            lufs = dbfs(rms)

        # noise floor: 10th percentile of 20ms window RMS; crest factor
        w = max(1, int(0.02 * sr)); nw = n // w
        wins = np.array([np.sqrt(np.mean(x[i*w:(i+1)*w] ** 2) + 1e-12) for i in range(nw)]) if nw else np.array([rms])
        noise_floor = dbfs(float(np.quantile(wins, 0.10)))
        crest = dbfs(peak) - dbfs(rms)

        # spectrum (full-take magnitude via Welch-ish average of STFT power)
        S = np.abs(librosa.stft(x, n_fft=4096, hop_length=1024)) ** 2
        psd = S.mean(axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
        total = float(psd.sum()) + 1e-12

        def band(lo, hi):
            m = (freqs >= lo) & (freqs < hi)
            e = float(psd[m].sum())
            return {"pct": round(100 * e / total, 2), "db": round(10 * np.log10(e / total + 1e-12), 1)}

        bands = {
            "sub_20_60": band(20, 60),
            "low_60_120": band(60, 120),
            "lowmid_120_300": band(120, 300),
            "mud_300_500": band(300, 500),
            "mid_500_2k": band(500, 2000),
            "presence_2k_5k": band(2000, 5000),
            "brilliance_5k_9k": band(5000, 9000),
            "air_9k_20k": band(9000, min(20000, sr // 2 - 1)),
        }

        centroid = float(np.mean(librosa.feature.spectral_centroid(y=x, sr=sr)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=x, sr=sr, roll_percent=0.85)))

        # sibilance index: 5-9k energy vs 1-3k energy (dB)
        def e(lo, hi):
            m = (freqs >= lo) & (freqs < hi); return float(psd[m].sum()) + 1e-12
        sibilance_db = round(10 * np.log10(e(5000, 9000) / e(1000, 3000)), 1)
        # mud index: 200-500 vs 500-2k
        mud_db = round(10 * np.log10(e(200, 500) / e(500, 2000)), 1)

        # mains hum: energy at 50/60 Hz +/-2Hz vs neighbourhood
        def line_ratio(f):
            m = (freqs >= f-2) & (freqs <= f+2)
            nb = (freqs >= f-15) & (freqs <= f+15)
            return round(10 * np.log10((psd[m].sum()+1e-12) / (psd[nb].mean()*m.sum()+1e-12)), 1)
        hum = {"50hz": line_ratio(50), "60hz": line_ratio(60)}

        # pitch / key
        f0, vflag, _ = librosa.pyin(x, sr=sr, fmin=librosa.note_to_hz("C2"),
                                    fmax=librosa.note_to_hz("C6"),
                                    frame_length=2048, hop_length=512)
        f0v = f0[np.isfinite(f0)] if f0 is not None else np.array([])
        if f0v.size:
            f0_med = float(np.median(f0v)); f0_lo = float(np.percentile(f0v, 5)); f0_hi = float(np.percentile(f0v, 95))
            note_lo = librosa.hz_to_note(f0_lo); note_hi = librosa.hz_to_note(f0_hi); note_med = librosa.hz_to_note(f0_med)
            voiced_pct = round(100 * float(np.mean(vflag)) if vflag is not None else 0.0, 1)
        else:
            f0_med = f0_lo = f0_hi = 0.0; note_lo = note_hi = note_med = "n/a"; voiced_pct = 0.0

        # key estimate via chroma (Krumhansl major/minor correlation)
        chroma = librosa.feature.chroma_cqt(y=x, sr=sr).mean(axis=1)
        names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        maj = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
        minr= np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
        best = None
        for i in range(12):
            for mode, prof in (("major", maj), ("minor", minr)):
                c = float(np.corrcoef(np.roll(prof, i), chroma)[0, 1])
                if best is None or c > best[2]:
                    best = (names[i], mode, c)
        key_guess = {"key": best[0], "scale": best[1], "confidence": round(best[2], 2)}

        print(json.dumps({
            "ok": True, "file": a.input,
            "sr": int(sr), "channels": int(ch), "duration_s": round(dur, 2),
            "peak_dbfs": round(dbfs(peak), 2), "rms_dbfs": round(dbfs(rms), 2),
            "integrated_lufs": round(lufs, 2), "crest_db": round(crest, 2),
            "noise_floor_dbfs": round(noise_floor, 2), "dc_offset": round(dc, 6),
            "clipped_samples": clip,
            "spectral_centroid_hz": round(centroid, 0), "rolloff85_hz": round(rolloff, 0),
            "sibilance_index_db": sibilance_db, "mud_index_db": mud_db, "mains_hum_db": hum,
            "bands": bands,
            "pitch": {"f0_median_hz": round(f0_med, 1), "median_note": note_med,
                       "low_note": note_lo, "high_note": note_hi, "voiced_pct": voiced_pct},
            "key_guess": key_guess,
        }))
    except Exception as ex:
        print(json.dumps({"ok": False, "error": f"{type(ex).__name__}: {ex}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
