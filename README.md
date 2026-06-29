# ComfyUI-VoiceCloneUI

A clean, standalone **voice-cloning web UI** for [ComfyUI-Qwen-TTS](https://github.com/flybirdxx/ComfyUI-Qwen-TTS), served directly by ComfyUI at **`/voiceclone`**. Windows 11 Fluent design, every cloning setting exposed, in-browser mic recording, live progress, instant playback & download — all running locally on your own GPU.

It drives the `FB_Qwen3TTSVoiceClone` node through ComfyUI's API, so there's no node graph to wire up.

## Features

- 🎙️ **Reference voice** by drag-and-drop file (any format) **or** in-browser microphone recording — auto-decoded to clean mono 16-bit WAV
- 📝 Reference transcript (`ref_text`) + target text, **11 languages**
- 🎛️ Every `FB_Qwen3TTSVoiceClone` setting: model quality (0.6B / 1.7B), temperature, top-p, top-k, repetition penalty, max-new-tokens, seed (+ randomize), x-vector-only, attention, precision, device, unload-after-generate
- ⚡ Live progress over ComfyUI's websocket; inline audio player, download, and an in-session history
- 🪟 Windows 11 Fluent styling (Segoe UI Variable, acrylic cards, WinUI controls), auto light/dark + manual toggle
- 🔒 Served **same-origin** by ComfyUI — no CORS setup, no extra server, no API keys; everything stays on your machine

## Requirements

- A working [ComfyUI](https://github.com/comfyanonymous/ComfyUI) install
- [ComfyUI-Qwen-TTS](https://github.com/flybirdxx/ComfyUI-Qwen-TTS) installed (provides the `FB_Qwen3TTSVoiceClone` node)
- The Qwen3-TTS **Base** model (`Qwen3-TTS-12Hz-1.7B-Base` or `0.6B-Base`) — auto-downloads on first run into `ComfyUI/models/qwen-tts/`

## Install

Clone into your ComfyUI `custom_nodes` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<your-user>/ComfyUI-VoiceCloneUI.git
```

Restart ComfyUI, then open:

```
http://127.0.0.1:8188/voiceclone
```

> **Windows note:** ComfyUI-Qwen-TTS prints emoji at startup. On a non-UTF-8 Windows console this can crash ComfyUI startup, so launch ComfyUI with `PYTHONUTF8=1` set.

## Usage

1. **Add a reference voice** — drop an audio file or record 5–15s from your mic.
2. **Type the transcript** of that clip (optional but improves accuracy) and **what you want it to say**.
3. Click **Generate cloned voice**. The result plays inline and can be downloaded; files are also saved to `ComfyUI/output/`.

## How it works

The page registers three aiohttp routes on ComfyUI's `PromptServer`:

| Route | Purpose |
|-------|---------|
| `GET /voiceclone` | serves the UI |
| `GET /voiceclone/status` | reports which Qwen-TTS models are downloaded |
| `POST /voiceclone/upload` | saves the reference clip into ComfyUI's `input/` folder |

On **Generate**, it builds an API prompt graph `LoadAudio → FB_Qwen3TTSVoiceClone → SaveAudio`, submits it to `/prompt`, tracks execution over `/ws`, and fetches the result from `/view`.

## License

MIT — see [LICENSE](LICENSE).

## Credits

- [ComfyUI-Qwen-TTS](https://github.com/flybirdxx/ComfyUI-Qwen-TTS) by flybirdxx, based on Alibaba's Qwen3-TTS
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) by comfyanonymous
