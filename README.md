# 🎬 AI Video Assistant

> Turn any video or meeting recording into a transcript, a structured summary and action items — then **chat with it** and ask questions grounded in what was actually said.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)
![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-orange)
![Whisper](https://img.shields.io/badge/STT-Whisper-purple)

<!-- Add a screenshot or GIF of the app here, e.g. ![Demo](assets/demo.gif) -->

## Table of contents
- [Features](#features)
- [How it works](#how-it-works)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Usage](#usage)
- [Configuration](#configuration)
- [Design decisions](#design-decisions)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)

## Features

- **Flexible input** — paste a YouTube URL (downloaded with `yt-dlp`) or point to any local audio/video file (converted with `pydub` + FFmpeg).
- **Speech-to-text in two languages**
  - English → OpenAI Whisper, running locally.
  - Hinglish → Sarvam AI speech-to-text-translate API, which transcribes and translates to English in one step.
- **Summaries that scale to long videos** — map-reduce: the transcript is split into chunks, each is summarised, and the partial summaries are merged into one bullet-point summary.
- **Structured extraction** — a short title, action items (task / owner / deadline), key decisions and open questions.
- **Chat with your video (RAG)** — ask free-form questions; answers come only from the transcript, and the assistant says so when the answer isn't there.
- **Choice of AI provider** — Google Gemini, Groq or Mistral, picked in the sidebar. All three have free tiers, so no paid key is needed.
- **Bring your own key** — API keys are typed into the sidebar and kept in the browser session only.
- **Resilient to free-tier limits** — requests are paced, rate limits are retried with backoff, retired Gemini models are skipped automatically, and a transcript is reused if you retry the same video.
- **Two interfaces** — a Streamlit web app with live pipeline status, and a command-line interface.

## How it works

```
YouTube URL / local file
        │
        ▼
 yt-dlp / pydub ──► 16 kHz mono WAV ──► 10-minute chunks
        │
        ▼
 Whisper (English)  or  Sarvam AI (Hinglish → English)
        │
        ▼
    Transcript
        ├─► Map-reduce summariser (LLM) ───────► Title + Summary
        ├─► Extractors (LLM) ──────────────────► Action items / Decisions / Questions
        └─► Splitter ─► MiniLM embeddings ─► FAISS ─► Retriever (top 4) ─► LLM ─► Answer

LLM = Google Gemini, Groq or Mistral — selectable in the sidebar
```

1. **Audio preparation** — video is converted to 16 kHz mono WAV and cut into 10-minute chunks so long recordings don't exhaust memory.
2. **Transcription** — each chunk is transcribed and the results are joined.
3. **Summarisation** — transcript is split into ~3000-character chunks (200 overlap), summarised individually, then combined.
4. **Extraction** — separate focused prompts pull out action items, decisions and open questions.
5. **RAG** — transcript is split into 500-character chunks (50 overlap), embedded with `all-MiniLM-L6-v2` and indexed in an in-memory FAISS vector store built fresh for every video. Each question retrieves the 4 most similar chunks, which are passed to the LLM with a prompt that restricts it to that context.

## Tech stack

| Area | Tools |
|---|---|
| Speech-to-text | OpenAI Whisper, Sarvam AI |
| LLM | Google Gemini (default), Groq or Mistral, via LangChain — choose in the sidebar |
| Orchestration | LangChain Expression Language (LCEL) |
| Retrieval | FAISS, sentence-transformers (`all-MiniLM-L6-v2`) |
| Audio | yt-dlp, pydub, FFmpeg |
| UI | Streamlit |

## Project structure

```
├── app.py                  # Streamlit web app
├── main.py                 # CLI entry point + run_pipeline()
├── test.py                 # Quick end-to-end script (no chat)
├── core/
│   ├── transcriber.py      # Whisper / Sarvam routing, chunked transcription
│   ├── llm.py              # LLM provider selection, request pacing, retry on rate limits
│   ├── summarizer.py       # Map-reduce summary + title generation
│   ├── extractor.py        # Action items, decisions, open questions
│   ├── vector_store.py     # Chunking, embeddings, FAISS index
│   └── rag_engine.py       # LCEL retrieval-augmented QA chain
├── utils/
│   └── audio_processor.py  # Download, convert and chunk audio
└── Requirements.txt
```

## Getting started

**Prerequisites**
- Python 3.10+ (developed on 3.12)
- [FFmpeg](https://ffmpeg.org/download.html) installed and on your `PATH` (Windows: `winget install Gyan.FFmpeg`, then open a new terminal)
- A free API key for **one** of: [Google Gemini](https://aistudio.google.com/apikey) (recommended, best for long videos), [Groq](https://console.groq.com/) or [Mistral](https://console.mistral.ai/)
- (Optional) A [Sarvam AI key](https://www.sarvam.ai/) for Hinglish videos

**Install**

```bash
git clone https://github.com/Sowjanya45/ai-video-assistant.git
cd ai-video-assistant
python -m venv venv
venv\Scripts\activate            # Windows PowerShell / cmd
source venv/Scripts/activate     # Windows Git Bash
source venv/bin/activate         # macOS / Linux
pip install -r Requirements.txt
```

The install is large (PyTorch and Whisper), so allow a few minutes. The first run also downloads the Whisper model (about 460 MB for `small`) and the embedding model once.

**API keys** — in the web app you enter them in the sidebar, so there is nothing to configure before the first run. Keys are held in your browser session's memory only and are never written to disk.

For the command-line version (or to avoid retyping keys locally), you can optionally create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_key      # or GROQ_API_KEY / MISTRAL_API_KEY
SARVAM_API_KEY=your_sarvam_key      # only for Hinglish
WHISPER_MODEL=small
```

## Usage

**Web app**

```bash
streamlit run app.py
```

In the sidebar, paste a YouTube URL or a local file path, choose a language, pick an AI provider and enter its API key (plus a Sarvam key if you chose Hinglish), then click **Analyse**. When the pipeline finishes you'll see the title, summary, action items, decisions and open questions, and a chat box to ask questions about the video.

**Command line**

```bash
python main.py
```

## Configuration

In the web app, API keys are entered in the sidebar. The environment variables below are an optional fallback, and are what the CLI uses. A key typed in the sidebar takes priority over `.env`.

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` / `GROQ_API_KEY` / `MISTRAL_API_KEY` | One of them (sidebar or `.env`) | Key for the chosen AI provider; used for summaries, extraction and chat answers |
| `LLM_PROVIDER` | No | `gemini`, `groq` or `mistral`. Only used by the CLI; the web app uses the sidebar choice |
| `SARVAM_API_KEY` | Only for Hinglish (sidebar or `.env`) | Sarvam speech-to-text-translate |
| `WHISPER_MODEL` | No (default `small`) | `tiny`, `base`, `small`, `medium` or `large` — larger is more accurate but slower |
| `SARVAM_STT_MODEL` | No (default `saaras:v2.5`) | Sarvam model name |
| `GEMINI_MODEL` | No (default `gemini-3.5-flash-lite`) | Pin a specific Gemini model instead of the built-in fallback list |

## Design decisions

- **Local Whisper for English** — free, private, no per-minute cost. Sarvam is used only where it adds value (Hinglish).
- **Map-reduce summarisation** — a long transcript doesn't fit in one prompt; summarising chunks first keeps cost and context size bounded.
- **Grounded answers** — the RAG prompt forbids answering outside the retrieved context to reduce hallucination.
- **FAISS for retrieval** — a single video's transcript is a few hundred chunks, so a lightweight in-memory FAISS index is fast and avoids running a separate database. (I originally used ChromaDB, but its native engine crashed on Windows, so I moved to FAISS.)
- **Fresh vector store per video** — each analysis gets its own in-memory collection, so one video's content can never leak into another's answers.
- **Bring your own key** — keys are passed directly into the functions that need them instead of being set as process-wide environment variables, so on a shared deployment one user's key can never be used by another user's session.
- **Escaped output** — all model/transcript text is HTML-escaped before it is rendered in the UI.
- **Cache the slow step** — transcription takes minutes on a CPU, so its result is kept in the session. If a later step fails (for example an API error), retrying the same video skips straight to the AI steps.
- **Handling free-tier limits** — the LangChain Mistral client doesn't retry HTTP 429 responses, so a shared wrapper in `core/llm.py` spaces requests per provider, retries rate limits with exponential backoff, and fails fast on bad keys.

## Limitations

- Transcription on CPU is slow for long videos; use a smaller `WHISPER_MODEL` or a GPU.
- Chat history isn't passed to the model, so each question is answered independently.
- The vector store is in memory: chat is lost when the app restarts.
- LLM requests are paced to stay under free-tier limits (one every ~2–7 seconds depending on the provider) and automatically retried on rate limits, so summaries of long videos take a while.
- Groq's free tier caps tokens per minute, so long videos can fail there; use Gemini for those.
- Free-tier model names get retired often. Gemini defaults to a Flash-Lite model (far higher free daily limits than full Flash) and automatically falls back to the next model if one is unavailable. You can pin a model with `GEMINI_MODEL` in `.env`.
- Transcripts have no timestamps yet.
- Quality depends on audio quality and on the YouTube download working (some videos are restricted).

## Roadmap

- [ ] Timestamps in transcript and in chat answers
- [ ] Conversation memory for follow-up questions
- [ ] Export summary and action items as PDF
- [ ] Speaker diarisation
- [ ] Docker image and hosted demo

## Troubleshooting

| Problem | Fix |
|---|---|
| `ffmpeg not found` / audio conversion errors | Install FFmpeg and make sure `ffmpeg -version` works in a new terminal |
| `401` / "invalid API key" errors | Re-enter the key for the provider selected in the sidebar and check it is active |
| Repeated "Rate limit hit, retrying" messages | The free quota is exhausted or throttled. Wait a few minutes, or switch provider in the sidebar |
| "Hinglish needs a Sarvam API key" | Enter the Sarvam key in the sidebar, or switch the language to English |
| Page is blank for a minute after `streamlit run` | Normal on the first load while PyTorch and Whisper import. Wait for it |
| First run is slow | Whisper and the embedding model are downloaded once on first use |
| Nothing prints for several minutes during transcription | Normal. Whisper prints nothing while it works, and CPU transcription runs at roughly real-time speed. Try `WHISPER_MODEL=base` or `tiny` to go faster |
| `Unsupported URL` | The source box must contain a link to the video (or a file path), not the app's own `localhost` address |
| YouTube download fails | Update yt-dlp: `pip install -U yt-dlp` |
| `404 NOT_FOUND ... model` from Gemini | Google retired that model. The app falls back automatically; if all fail, set `GEMINI_MODEL` in `.env` to a current model name |
| `OSError: [WinError 1114] ... c10.dll` when importing torch (Windows) | Install a known-good CPU build: `pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu` |
| Long `Examining the path of transformers...` warnings in the terminal | Harmless. Silence them with `streamlit run app.py --server.fileWatcherType none` |
| `FP16 is not supported on CPU` warning | Harmless. Whisper falls back to normal precision on CPUs |
