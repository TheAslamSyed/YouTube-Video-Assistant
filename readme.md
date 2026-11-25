# AI Video Assistant – Transcribe, Summarize and Chat with Any Video

**A Generative AI application that turns a YouTube video or meeting recording into a transcript, a structured summary, action items, key decisions and open questions. You can then chat with the video using Retrieval-Augmented Generation (RAG), and every answer comes only from what was actually said.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/LangChain-LCEL-green" />
  <img src="https://img.shields.io/badge/RAG-FAISS-orange" />
  <img src="https://img.shields.io/badge/Speech--to--Text-Whisper-purple" />
  <img src="https://img.shields.io/badge/UI-Streamlit-ff4b4b?logo=streamlit&logoColor=white" />
</p>

<!-- Add a screenshot or GIF of the app here, for example: ![Demo](assets/demo.gif) -->

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Features](#2-key-features)
3. [Tools and Technologies](#3-tools-and-technologies)
4. [System Architecture](#4-system-architecture)
5. [Pipeline Walkthrough](#5-pipeline-walkthrough)
6. [Chat with the Video (RAG)](#6-chat-with-the-video-rag)
7. [Handling Free-Tier API Limits](#7-handling-free-tier-api-limits)
8. [Design Decisions](#8-design-decisions)
9. [Project Structure](#9-project-structure)
10. [How to Run the Project](#10-how-to-run-the-project)
11. [Configuration](#11-configuration)
12. [Troubleshooting](#12-troubleshooting)
13. [Limitations and Future Improvements](#13-limitations-and-future-improvements)
14. [Author](#14-author)

---

## 1. Project Overview

Long videos and recorded meetings hold useful information, but finding it means watching the whole thing again. It is hard to remember who agreed to do what, what was decided, and which questions were left open.

This project builds an **end-to-end AI pipeline** that does this work automatically:

1. Takes a **YouTube URL** or a **local audio/video file**.
2. Converts speech to text using **OpenAI Whisper** (English) or **Sarvam AI** (Hinglish, translated to English).
3. Uses a **Large Language Model (LLM)** to generate a title, a bullet-point summary, action items, key decisions and open questions.
4. Builds a **FAISS vector index** of the transcript so you can **ask questions about the video** and get answers grounded in the transcript.

The application runs as a **Streamlit web app** with live pipeline status, and also as a **command-line tool**.

**Free to run:** Whisper runs locally, and the LLM can be Google Gemini, Groq or Mistral. All three have free tiers.

## 2. Key Features

| Feature | Description |
|---|---|
| Flexible input | YouTube URLs (downloaded with `yt-dlp`) or any local audio/video file (converted with `pydub` + FFmpeg) |
| Two transcription languages | English → Whisper running locally; Hinglish → Sarvam AI speech-to-text-translate |
| Scales to long videos | Audio is processed in 10-minute chunks; the summary uses a map-reduce approach |
| Structured extraction | Title, summary, action items (task / owner / deadline), key decisions and open questions |
| Chat with your video | RAG-based question answering that only uses the transcript and says so when the answer is not there |
| Choice of AI provider | Google Gemini, Groq or Mistral, selected in the sidebar |
| Bring your own key | API keys are entered in the sidebar and kept only in the browser session |
| Resilient to rate limits | Request pacing, retries with exponential backoff, and automatic fallback when a Gemini model is retired |
| Transcript caching | If a later step fails, retrying the same video skips the slow transcription step |
| Two interfaces | Streamlit web app and command-line interface |

## 3. Tools and Technologies

| Category | Tools |
|---|---|
| Programming Language | Python |
| Speech-to-Text | OpenAI Whisper (local), Sarvam AI API |
| Large Language Models | Google Gemini, Groq (Llama 3.1), Mistral |
| LLM Orchestration | LangChain, LangChain Expression Language (LCEL) |
| Embeddings | Sentence-Transformers `all-MiniLM-L6-v2` (Hugging Face) |
| Vector Search | FAISS (in-memory) |
| Audio Processing | yt-dlp, pydub, FFmpeg |
| Deep Learning Backend | PyTorch |
| User Interface | Streamlit |
| Configuration | python-dotenv |
| Version Control | Git, GitHub |

## 4. System Architecture

```text
            YouTube URL  /  Local audio or video file
                              │
                              ▼
               yt-dlp  /  pydub + FFmpeg  →  WAV audio
                              │
                              ▼
                     Split into 10-minute chunks
                              │
                              ▼
      Whisper (English)   or   Sarvam AI (Hinglish → English)
                              │
                              ▼
                         Full transcript
                              │
         ┌────────────────────┼─────────────────────────┐
         ▼                    ▼                         ▼
  Map-reduce summary    Extraction prompts       Text splitter (500 chars)
      + title           (action items,                  │
         │               decisions, questions)          ▼
         │                    │                 MiniLM embeddings → FAISS
         │                    │                         │
         ▼                    ▼                         ▼
   Title + Summary     Structured insights     Retriever (top 4) → LLM → Answer

        LLM = Google Gemini, Groq or Mistral (selected in the sidebar)
```

## 5. Pipeline Walkthrough

### Step 1: Audio Preparation (`utils/audio_processor.py`)

- **YouTube URL:** the best audio stream is downloaded with `yt-dlp` and extracted to WAV using FFmpeg.
- **Local file:** any audio or video format is converted to **16 kHz mono WAV** using `pydub`.
- The audio is cut into **10-minute chunks**, so long recordings do not run out of memory.

### Step 2: Transcription (`core/transcriber.py`)

| Language | Engine | How it works |
|---|---|---|
| English | OpenAI Whisper | Runs locally on your machine. The model size (`tiny` to `large`) is configurable. The model is loaded once and reused. |
| Hinglish | Sarvam AI | Sarvam's speech-to-text-translate API transcribes Hindi-English speech and translates it to English in one step. Its sync API accepts at most 30 seconds of audio, so each chunk is split into **25-second pieces**. |

The text from all chunks is joined into one transcript.

### Step 3: Title and Summary (`core/summarizer.py`)

A long transcript does not fit into one LLM prompt, so the summary uses a **map-reduce** approach:

1. **Split:** the transcript is split into ~3,000-character chunks with 200 characters of overlap.
2. **Map:** each chunk is summarized separately.
3. **Reduce:** the partial summaries are combined into one final bullet-point summary.

The **title** (8 words or fewer) is generated from the first 2,000 characters of the transcript.

### Step 4: Structured Extraction (`core/extractor.py`)

Three focused prompts run on the transcript at a low temperature (0.2) for consistent output:

| Output | What is extracted |
|---|---|
| Action items | Task description, owner, and deadline ("Not specified" if not mentioned) |
| Key decisions | Decisions made during the discussion |
| Open questions | Unresolved questions and topics that need follow-up |

### Step 5: Build the RAG Chain (`core/vector_store.py`, `core/rag_engine.py`)

The transcript is indexed for question answering. This is explained in the next section.

### Pipeline Parameters

| Parameter | Value |
|---|---|
| Audio chunk length | 10 minutes |
| Sarvam piece length | 25 seconds |
| Summary chunk size / overlap | 3,000 / 200 characters |
| Title input | First 2,000 characters |
| RAG chunk size / overlap | 500 / 50 characters |
| Retrieved chunks per question | 4 |
| LLM temperature | 0.3 (summary, title, chat), 0.2 (extraction) |
| Embedding model | `all-MiniLM-L6-v2` |

## 6. Chat with the Video (RAG)

Retrieval-Augmented Generation (RAG) lets the LLM answer questions using only the relevant parts of the transcript, instead of relying on its own general knowledge.

### How a question is answered

1. **Chunking:** the transcript is split into 500-character chunks with 50 characters of overlap.
2. **Embedding:** each chunk is converted into a vector using the `all-MiniLM-L6-v2` sentence-transformer model (runs on CPU).
3. **Indexing:** the vectors are stored in an **in-memory FAISS index**, built fresh for every video.
4. **Retrieval:** for each question, the **4 most similar chunks** are retrieved.
5. **Generation:** the chunks and the question are passed to the LLM with a prompt that restricts it to that context.

### Grounded answers

The system prompt tells the model to answer **only** from the retrieved transcript context. If the answer is not there, it replies:

> "I could not find this information in the meeting transcript."

This reduces hallucination, so the assistant does not invent facts that were never said in the video.

### LCEL chain

The full RAG pipeline is written in LangChain Expression Language:

```python
rag_chain = (
    {"context": retriever | RunnableLambda(format_docs),
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

## 7. Handling Free-Tier API Limits

Free LLM tiers limit requests per minute and per day, and a single video needs several LLM calls (title + one call per summary chunk + final summary + 3 extractions). Without special handling, a single rate-limit error would stop the whole pipeline.

`core/llm.py` contains a shared wrapper that handles this for all three providers:

| Problem | Solution |
|---|---|
| Too many requests too quickly | One shared rate limiter per provider spaces out all calls (Gemini: 1 every 6 s, Groq: 2.5 s, Mistral: 2 s) |
| HTTP 429 / quota exceeded | Retries up to 5 times with exponential backoff (4 s, 8 s, 16 s … capped at 60 s, plus random jitter) |
| Some clients do not retry 429s (Mistral) | The wrapper adds retry logic on top of every provider's chat model |
| Gemini model names are retired often | If a model returns "not found", the wrapper automatically switches to the next model in a fallback list |
| Wrong or missing API key | Fails fast with a clear error message instead of retrying |

### Default Models

| Provider | Default model | Override with |
|---|---|---|
| Google Gemini | First model in the fallback list in `core/llm.py` (a Flash-Lite model) | `GEMINI_MODEL` |
| Groq | `llama-3.1-8b-instant` | `GROQ_MODEL` |
| Mistral | `mistral-small-latest` | `MISTRAL_MODEL` |

Gemini defaults to a **Flash-Lite** model because its free tier allows far more requests per day than the full Flash models.

## 8. Design Decisions

| Decision | Reason |
|---|---|
| Local Whisper for English | Free, private, and no per-minute cost. Sarvam is used only where it adds value (Hinglish). |
| Map-reduce summarization | Keeps each prompt small enough for the model's context window and free-tier token limits. |
| Grounded RAG prompt | Answers come only from the transcript, which reduces hallucination. |
| FAISS instead of a vector database | A single video produces only a few hundred chunks, so an in-memory index is fast and needs no separate server. ChromaDB was used at first, but its native engine crashed on Windows. |
| Fresh vector store per video | One video's content can never leak into another video's answers. |
| Bring your own key | Keys are passed directly to the functions that need them, not set as global environment variables. On a shared deployment, one user's key can never be used by another user's session. |
| Escaped output | All model and transcript text is HTML-escaped before it is shown in the UI. |
| Transcript caching | Transcription takes minutes on a CPU, so the result is kept in the session. If a later step fails, a retry skips straight to the AI steps. |
| Lazy provider imports | Only the selected provider's LangChain package is loaded. |

## 9. Project Structure

```text
YouTube-Video-Assistant/
│
├── core/
│   ├── transcriber.py      # Whisper / Sarvam routing, chunked transcription
│   ├── llm.py              # LLM provider selection, rate limiting, retries, model fallback
│   ├── summarizer.py       # Map-reduce summary and title generation
│   ├── extractor.py        # Action items, key decisions, open questions
│   ├── vector_store.py     # Text chunking, embeddings, FAISS index
│   └── rag_engine.py       # LCEL retrieval-augmented QA chain
│
├── utils/
│   └── audio_processor.py  # Download, convert and chunk audio
│
├── app.py                  # Streamlit web app
├── main.py                 # Command-line interface and run_pipeline()
├── test.py                 # Quick end-to-end test script (no chat)
├── Requirements.txt
└── README.md
```

Downloaded and converted audio files are saved in a `downloads/` folder, which is created automatically.

## 10. How to Run the Project

### Prerequisites

- **Python 3.10+** (developed on Python 3.12)
- **FFmpeg** installed and available on your `PATH`
  - Windows: `winget install Gyan.FFmpeg`, then open a new terminal
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
- A free API key for **one** of these:
  - [Google Gemini](https://aistudio.google.com/apikey) (recommended, best for long videos)
  - [Groq](https://console.groq.com/)
  - [Mistral](https://console.mistral.ai/)
- (Optional) A [Sarvam AI](https://www.sarvam.ai/) API key for Hinglish videos

### Step 1: Clone the repository

```bash
git clone https://github.com/TheAslamSyed/YouTube-Video-Assistant.git

cd YouTube-Video-Assistant
```

### Step 2: Create and activate a virtual environment

```bash
python -m venv venv
```

```bash
venv\Scripts\activate            # Windows (PowerShell / cmd)
source venv/Scripts/activate     # Windows (Git Bash)
source venv/bin/activate         # macOS / Linux
```

### Step 3: Install dependencies

```bash
pip install -r Requirements.txt
```

The installation is large because it includes PyTorch and Whisper, so it can take a few minutes. On first use, the Whisper model (about 460 MB for `small`) and the embedding model are downloaded once.

### Step 4: Run the web app

```bash
streamlit run app.py
```

In the sidebar:

1. Paste a **YouTube URL** or a **local file path**.
2. Choose the **language** (`english` or `hinglish`).
3. Pick an **AI provider** and enter its **API key** (and a Sarvam key for Hinglish).
4. Click **Analyse**.

When the pipeline finishes, you will see the title, summary, full transcript, action items, key decisions and open questions, along with a chat box for asking questions about the video.

### Optional: Run from the command line

Create a `.env` file in the project root (see [Configuration](#11-configuration)), then run:

```bash
python main.py
```

Enter the URL or file path and the language when prompted. After the results are printed, you can chat with the video in the terminal. Type `exit` to quit.

## 11. Configuration

In the web app, API keys are entered in the sidebar, so nothing needs to be set up before the first run. Keys are kept only in the browser session's memory and are never written to disk.

For the command-line version, or to avoid retyping keys locally, create a `.env` file:

```env
GOOGLE_API_KEY=your_gemini_key      # or GROQ_API_KEY / MISTRAL_API_KEY
SARVAM_API_KEY=your_sarvam_key      # only for Hinglish
WHISPER_MODEL=small
```

A key typed in the sidebar takes priority over `.env`.

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` / `GROQ_API_KEY` / `MISTRAL_API_KEY` | One of them (sidebar or `.env`) | Key for the chosen AI provider |
| `LLM_PROVIDER` | No | `gemini`, `groq` or `mistral`. Used by the CLI only; the web app uses the sidebar choice |
| `SARVAM_API_KEY` | Only for Hinglish | Sarvam AI speech-to-text-translate key |
| `WHISPER_MODEL` | No (default `small`) | `tiny`, `base`, `small`, `medium` or `large`. Larger models are more accurate but slower |
| `SARVAM_STT_MODEL` | No (default `saaras:v2.5`) | Sarvam model name |
| `GEMINI_MODEL` | No | Pin a specific Gemini model instead of the built-in fallback list |
| `GROQ_MODEL` | No (default `llama-3.1-8b-instant`) | Groq model name |
| `MISTRAL_MODEL` | No (default `mistral-small-latest`) | Mistral model name |

**Never commit your `.env` file.** Add it to `.gitignore`.

## 12. Troubleshooting

| Problem | Fix |
|---|---|
| `ffmpeg not found` or audio conversion errors | Install FFmpeg and check that `ffmpeg -version` works in a new terminal |
| `401` / "invalid API key" | Re-enter the key for the provider selected in the sidebar and check that it is active |
| Repeated "Rate limit hit, retrying" messages | The free quota is exhausted or throttled. Wait a few minutes or switch provider |
| "Hinglish needs a Sarvam API key" | Enter the Sarvam key in the sidebar, or switch the language to English |
| Page is blank for a minute after `streamlit run` | Normal on first load while PyTorch and Whisper are imported |
| Nothing prints for several minutes during transcription | Normal. CPU transcription runs at roughly real-time speed. Use `WHISPER_MODEL=base` or `tiny` for faster results |
| `Unsupported URL` | The input box must contain a video link or file path, not the app's own `localhost` address |
| YouTube download fails | Update yt-dlp: `pip install -U yt-dlp` |
| `404 NOT_FOUND ... model` from Gemini | That model was retired. The app falls back automatically; if all fail, set `GEMINI_MODEL` to a current model name |
| `OSError: [WinError 1114] ... c10.dll` when importing torch (Windows) | Install a CPU build: `pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu` |
| Long `Examining the path of transformers...` warnings | Harmless. Silence them with `streamlit run app.py --server.fileWatcherType none` |
| `FP16 is not supported on CPU` warning | Harmless. Whisper falls back to normal precision on CPUs |

## 13. Limitations and Future Improvements

### Current Limitations

- Transcription on a CPU is slow for long videos. A GPU or a smaller Whisper model speeds it up.
- Chat history is not sent to the model, so each question is answered independently and follow-up questions such as "what about the second one?" do not work.
- The vector store is in memory, so the chat is lost when the app restarts.
- Requests are paced to stay within free-tier limits, so summaries of long videos take some time.
- Groq's free tier caps tokens per minute, so long videos may fail there. Gemini works better for them.
- Transcripts do not have timestamps yet.
- Output quality depends on audio quality, and some YouTube videos cannot be downloaded because of restrictions.
- Downloaded audio files in `downloads/` are not deleted automatically.

### Future Improvements

- [ ] Timestamps in the transcript and in chat answers, with links to that point in the video
- [ ] Conversation memory for follow-up questions
- [ ] Export the summary and action items as PDF or TXT
- [ ] Speaker diarization (who said what)
- [ ] GPU support detection for faster transcription
- [ ] Automatic cleanup of temporary audio files
- [ ] Evaluation of RAG answer quality
- [ ] Docker image and hosted demo

## 14. Author

**Syed Abdul Aslam Shareef**

B.Tech, National Institute of Technology Warangal

- **GitHub:** [@TheAslamSyed](https://github.com/TheAslamSyed)
- **LinkedIn:** [Aslam Syed](https://www.linkedin.com/in/aslam-syed-8a9b7b303)

---

*This project demonstrates practical skills in speech-to-text, Large Language Models, prompt engineering, LangChain (LCEL), Retrieval-Augmented Generation, vector search with FAISS, handling API rate limits, and building an interactive AI application with Streamlit.*

If you found this project useful, consider giving it a ⭐
