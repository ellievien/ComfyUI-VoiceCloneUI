# ComfyUI-VoiceCloneUI

A clean, standalone **voice-cloning web UI + vocal "Studio Process" backend** for
[ComfyUI-Qwen-TTS](https://github.com/flybirdxx/ComfyUI-Qwen-TTS), served directly by
ComfyUI at **`/voiceclone`**. Windows 11 Fluent design, every cloning setting exposed,
in-browser mic recording, live progress, instant playback & download — all running
locally on your own machine.

It drives the `FB_Qwen3TTSVoiceClone` node through ComfyUI's API (no node graph to wire
up), and also exposes HTTP routes used by two companion VST3 plugins:
[**nvh suno polish**](https://github.com/ellievien/nvh-suno-polish) and **Qwen Voice Clone**.

## Features

- 🎙️ **Reference voice** by drag-and-drop file (any format) **or** in-browser microphone recording — auto-decoded to clean mono 16-bit WAV
- 🗣️ **Auto-transcribe** the reference clip (local Whisper) so you don't type `ref_text`
- 📝 Target text + reference transcript, **12 languages** (incl. **Tagalog**)
- 🎛️ Every `FB_Qwen3TTSVoiceClone` setting: model quality (0.6B / 1.7B), temperature, top-p, top-k, repetition penalty, max-new-tokens, seed (+ randomize), x-vector-only, attention, precision, device, unload-after-generate
- 🧰 **Studio Process** backend for the plugins: **Enhance** (DeepFilterNet restoration), **Regenerate** (clone in a stored "my voice"), and **Suno Polish** (clean → autotune → production EQ/comp/sat/width → reverb → loud master)
- ⚡ Live progress over ComfyUI's websocket; inline audio player, download, in-session history
- 🪟 Windows 11 Fluent styling (Segoe UI Variable, acrylic cards, WinUI controls), auto light/dark + manual toggle
- 🔒 Served **same-origin** by ComfyUI — no CORS setup, no extra server, no API keys

## Requirements

- A working [ComfyUI](https://github.com/comfyanonymous/ComfyUI) install
- [ComfyUI-Qwen-TTS](https://github.com/flybirdxx/ComfyUI-Qwen-TTS) (provides `FB_Qwen3TTSVoiceClone`)
- The Qwen3-TTS **Base** model — auto-downloads on first run into `ComfyUI/models/qwen-tts/`
- For **auto-transcribe**: `transformers` + a local `openai/whisper-small` (the node points at `ComfyUI/models/whisper-small`)
- For **Enhance / Suno Polish**: a separate **isolated Python 3.11 venv** with DeepFilterNet, pedalboard, librosa, psola, pyloudnorm — see [`studio_backend/`](studio_backend/). Kept isolated so it never disturbs ComfyUI's own torch/transformers.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ellievien/ComfyUI-VoiceCloneUI.git
```

Restart ComfyUI, then open `http://127.0.0.1:8188/voiceclone`.

> **Windows note:** ComfyUI-Qwen-TTS prints emoji at startup; on a non-UTF-8 Windows console this can crash ComfyUI startup, so launch ComfyUI with `PYTHONUTF8=1` set.

## Routes

| Route | Purpose |
|-------|---------|
| `GET  /voiceclone` | serves the web UI |
| `GET  /voiceclone/status` | which Qwen-TTS models are downloaded |
| `POST /voiceclone/upload` | save an uploaded clip into ComfyUI's `input/` |
| `POST /voiceclone/transcribe` | Whisper ASR of a clip (`{filename, language}` → `{text}`) |
| `POST /voiceclone/enhance` | DeepFilterNet restoration of a take (isolated venv) |
| `POST /voiceclone/polish` | **Suno Polish** chain on a take (isolated venv) |
| `GET/POST /voiceclone/myvoice` | get / store the "my voice" reference for Regenerate |

On **Generate**, the UI builds an API graph `LoadAudio → FB_Qwen3TTSVoiceClone → SaveAudio`,
submits it to `/prompt`, tracks execution over `/ws`, and fetches the result from `/view`.

`enhance` and `polish` shell out to the isolated venv scripts in [`studio_backend/`](studio_backend/)
(they do **not** import any ML into ComfyUI's process), write a WAV into `output/`, and return its
filename + metrics.

> **Language note:** Whisper transcribes all 12 languages (incl. Tagalog), but Qwen3-TTS only
> generates its supported set — so the UI/plugins send the TTS node `"Auto"` for any language the
> model doesn't list (e.g. Tagalog), while transcription still uses the chosen language.

## License

MIT — see [LICENSE](LICENSE).

## Credits

- [ComfyUI-Qwen-TTS](https://github.com/flybirdxx/ComfyUI-Qwen-TTS) by flybirdxx, based on Alibaba's Qwen3-TTS
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) by comfyanonymous
- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet), [pedalboard](https://github.com/spotify/pedalboard), [psola](https://github.com/maxrmorrison/psola) for the Studio Process backend
