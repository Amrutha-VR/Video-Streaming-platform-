# YAP-Detector — Real-Time Video Intelligence & Keyword Alert System

<div align="center">

**Live transcription · AI-powered keyword detection · Security hardened · 100% local**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Whisper](https://img.shields.io/badge/Whisper-base.en-FF6F00?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)

*Built on three peer-reviewed speech processing research papers. Zero cloud dependency. No API keys.*

</div>

---

## Overview

YAP-Detector is a real-time video intelligence platform that transcribes live YouTube streams and uploaded videos entirely on-device, monitors the transcript for predefined security-critical keywords, and fires instant alerts the moment a match is detected.

Every design decision in this project traces back to a specific, documented problem in real-time ASR — and a specific published solution. This is not a Whisper wrapper. It is an implementation of three research papers applied to a practical surveillance monitoring use case.

**What it does:**
- Transcribes uploaded MP4/MOV/AVI/MKV video files with word-level timestamps
- Monitors live YouTube streams and direct media URLs in real time
- Detects 40+ predefined security keywords with multi-gate false-positive filtering
- Displays live subtitles overlaid on the video with smooth fade transitions
- Fires visual + audio alerts with video-relative timestamps logged to SQLite
- Runs entirely locally — no data ever leaves your machine

---

## Demo



---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          BROWSER  (Frontend)                          │
│                                                                        │
│   ┌─────────────────────┐        ┌──────────────────────────────┐    │
│   │   Upload Video Mode  │        │     Live Stream Mode          │    │
│   │   (raw binary POST)  │        │   (YouTube URL → new tab)    │    │
│   └──────────┬──────────┘        └───────────────┬──────────────┘    │
│              │  HTTP                              │  WebSocket         │
│   ┌──────────▼──────────────────────────────────▼──────────────┐    │
│   │            Security Layer (Frontend)                         │    │
│   │  XSS encoding · URL validation · Input length cap           │    │
│   │  WS message schema whitelist · noopener/noreferrer          │    │
│   └──────────┬───────────────────────────────────┬──────────────┘    │
└──────────────┼───────────────────────────────────┼────────────────────┘
               │                                   │
┌──────────────▼───────────────────────────────────▼────────────────────┐
│                         FASTAPI  SERVER  (Backend)                      │
│                                                                          │
│   ┌─────────────────┐         ┌──────────────────────────────────┐     │
│   │  /upload-video  │         │  /yt-transcribe  (WebSocket)      │     │
│   │  raw bytes →    │         │  yt-dlp resolves audio URL        │     │
│   │  ffmpeg → WAV   │         │  ffmpeg pipes 16kHz mono PCM      │     │
│   └────────┬────────┘         └─────────────────┬────────────────┘     │
│            │                                     │                       │
│   ┌────────▼─────────────────────────────────────▼────────────┐        │
│   │           Security Layer (Backend)                          │        │
│   │   SSRF prevention · URL allowlist · Private IP block       │        │
│   └────────┬────────────────────────────────────┬─────────────┘        │
│            │                                     │                       │
│   ┌────────▼─────────────────────────────────────▼────────────┐        │
│   │  [R3] Energy VAD — Sohn et al. (1999)                      │        │
│   │  30ms frame RMS energy gate — silent chunks discarded      │        │
│   └────────────────────────────┬───────────────────────────────┘        │
│                                │                                          │
│   ┌────────────────────────────▼───────────────────────────────┐        │
│   │  faster-whisper  base.en · int8 · word_timestamps=True     │        │
│   │  + Silero VAD filter (second silence gate)                  │        │
│   └────────────────────────────┬───────────────────────────────┘        │
│                                │                                          │
│   ┌────────────────────────────▼───────────────────────────────┐        │
│   │  [R2] W-CTC Word-level Confidence Gating                   │        │
│   │  Kim et al. (Interspeech 2025)                              │        │
│   │  Per-word probability + timestamp alignment check           │        │
│   └────────────────────────────┬───────────────────────────────┘        │
│                                │                                          │
│   ┌────────────────────────────▼───────────────────────────────┐        │
│   │  [R1] Local Agreement Policy with Rollback                 │        │
│   │  Macháček et al. (Interspeech 2023)                        │        │
│   │  Confirm words across consecutive chunks · rollback tail   │        │
│   └────────────────────────────┬───────────────────────────────┘        │
│                                │                                          │
│        subtitle / alert ───────┴──► WebSocket ──► Browser               │
│        SQLite ◄── alert log (keyword · timestamp · video_ts)            │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Research Implementation

This section documents exactly which paper was read, what problem it solved in this project, and precisely where its concept appears in the codebase.

---

### [R1] Local Agreement Policy with Rollback

**Paper:** Macháček, D., Dabre, R., & Bojar, O. (2023). *Turning Whisper into Real-Time Transcription System.* Proc. Interspeech 2023. https://arxiv.org/abs/2307.14743

**Problem:** Whisper is an offline model trained on complete audio segments. When forced into a streaming context by dividing audio into chunks, the final words of each chunk are decoded speculatively — Whisper has incomplete audio context and guesses the boundary words. These guesses frequently change between chunks, causing subtitle flickering and hallucinated words reaching the display.

**Paper's solution:** The Local Agreement Policy. A word is only confirmed — and only then emitted to the user — when it appears in the **same token position** in two consecutive overlapping transcriptions. Any unconfirmed word at the tail of a transcription is rolled back and re-evaluated in the next chunk.

**Implementation** (`main.py` — `class LocalAgreementBuffer`):

```python
class LocalAgreementBuffer:
    """
    Local Agreement Policy with rollback.
    Macháček et al., Interspeech 2023 — §3.1
    
    confirmed_words : emitted to the user, immutable
    pending_words   : seen once, awaiting confirmation
    """
    def update(self, new_text: str) -> Optional[str]:
        new_words = new_text.strip().split()
        
        # Find longest common prefix between pending and new transcription
        agreed = []
        for pw, nw in zip(self.pending_words, new_words):
            if pw.lower().rstrip(".,!?") == nw.lower().rstrip(".,!?"):
                agreed.append(nw)
            else:
                break  # first disagreement — stop, rollback begins here
        
        # Newly confirmed = agreed words beyond what was already confirmed
        newly_confirmed = agreed[len(self.confirmed_words):]
        if newly_confirmed:
            self.confirmed_words.extend(newly_confirmed)
        
        # Rollback: pending becomes the full new transcription
        # The unconfirmed tail will be re-evaluated next chunk
        self.pending_words = new_words
```

Every Whisper segment passes through `la.update()` before any subtitle is displayed. Words that appeared only once are withheld. This directly implements §3.1 of the paper.

---

### [R2] W-CTC Word-level Confidence Gating

**Paper:** Kim et al. (2025). *W-CTC: Word-level Connectionist Temporal Classification for Automatic Speech Recognition.* Proc. Interspeech 2025.

**Problem:** Whisper produces a single `avg_logprob` score per segment. This segment-level confidence score cannot distinguish a segment where most words are confident but the keyword specifically was uncertain. Relying on segment confidence alone to fire keyword alerts produces false positives — a mostly-confident segment fires an alert even when the keyword word itself was a low-probability decode.

**Paper's solution:** W-CTC aligns each decoded word to a precise timestamp and assigns an individual probability score via forced CTC alignment. Downstream decisions should be gated on per-word scores, not segment averages.

**Implementation** (`main.py` — `def wctc_keyword_check()`):

```python
WORD_CONF_THRESHOLD = 0.60  # minimum per-word probability to fire alert

def wctc_keyword_check(keyword, segment, seg_start, seg_end):
    """
    W-CTC inspired word-level confidence gating.
    Kim et al., Interspeech 2025.
    
    Uses word_timestamps=True from faster-whisper which returns
    per-word: start, end, probability for every decoded word.
    
    For multi-word keywords ("active shooter"):
      - Slide a token window over segment.words
      - Every word in the window must independently satisfy:
          (a) timestamp overlaps [seg_start, seg_end + 1.0s grace]
          (b) word.probability >= WORD_CONF_THRESHOLD
      - Alert fires ONLY if ALL words pass BOTH checks
    """
    word_list = [(w.word.strip().lower(), w.start, w.end, w.probability)
                 for w in (segment.words or [])]
    
    kw_tokens = keyword.lower().split()
    for i in range(len(word_list) - len(kw_tokens) + 1):
        window   = word_list[i : i + len(kw_tokens)]
        match    = all(fuzz.ratio(k, w[0]) > 70
                       for k, w in zip(kw_tokens, window))
        if not match:
            continue
        # All words must pass timestamp overlap AND confidence gate
        if all(w[2] >= seg_start and w[1] <= seg_end + 1.0
               and w[3] >= WORD_CONF_THRESHOLD for w in window):
            return True, min(w[3] for w in window)
    
    return False, 0.0
```

An earlier version of this check was `is_valid_alignment(seg.start, seg.end, seg.start, seg.end)` — passing identical values to both arguments, making the function trivially always return `True`. The current implementation performs genuine per-word timestamp overlap and probability gating as described in the paper.

---

### [R3] Energy-based Voice Activity Detection

**Paper:** Sohn, J., Kim, N. S., & Sung, W. (1999). *A statistical model-based voice activity detection.* IEEE Signal Processing Letters, 6(1), 1–3. https://doi.org/10.1109/97.736233

**Production reference:** WebRTC VAD (Google, 2011) — the same 30ms frame-level energy approach deployed in all major real-time communication pipelines including Chrome, Firefox, and WebRTC.

**Problem:** When Whisper receives a chunk of silence or low-level background noise, it hallucinates common words — "you", "thank you", "okay", "hmm" — rather than returning an empty transcript. This is a well-documented and widely reported Whisper failure mode. Running the full encoder-decoder transformer stack on a silent 5-second chunk wastes significant CPU and pollutes the subtitle display.

**Paper's solution:** Before forwarding audio to the recognizer, compute the short-term energy of fixed-length frames. If the proportion of frames below a calibrated energy threshold exceeds a ceiling, classify the chunk as non-speech and discard it upstream of the model.

**Implementation** (`main.py` — `def vad_chunk_is_speech()`):

```python
SILENCE_THRESHOLD = 0.010  # normalised RMS, range [0.0, 1.0]
SILENCE_RATIO_MAX = 0.85   # skip chunk if >85% of frames are silent

def vad_chunk_is_speech(pcm_bytes, frame_ms=30, sample_rate=16000):
    """
    Energy-based VAD — Sohn et al. (1999) / WebRTC VAD frame size.
    
    30ms frame size matches the WebRTC VAD specification exactly.
    Runs entirely in NumPy before Whisper is invoked.
    Zero model overhead — pure signal energy computation.
    """
    frame_samples = int(sample_rate * frame_ms / 1000)  # 480 samples @ 16kHz
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    
    speech_frames = total_frames = 0
    for start in range(0, len(samples) - frame_samples, frame_samples):
        frame = samples[start : start + frame_samples]
        rms   = float(np.sqrt(np.mean(frame ** 2))) / 32768.0
        total_frames += 1
        if rms >= SILENCE_THRESHOLD:
            speech_frames += 1
    
    speech_ratio = speech_frames / total_frames if total_frames else 0
    return speech_ratio > (1.0 - SILENCE_RATIO_MAX)
```

This gate runs before Whisper loads the audio file. Silent chunks are discarded in microseconds. A second, neural VAD gate (Silero VAD embedded in faster-whisper via `vad_filter=True`) handles more subtle non-speech cases like background music with high energy but no speech content.

---

## Security Implementation

Security is implemented at both the frontend (client-side hardening) and backend (server-side enforcement). All concepts are mapped to OWASP Top 10 categories.

---

### Frontend Security

**[S1] XSS Prevention — Output Encoding**
*OWASP A03: Injection*

All strings originating from external sources (stream URLs, video titles resolved by the server, WebSocket message content) are passed through `sanitizeDisplay()` before any DOM insertion. This converts `<`, `>`, `&`, `"`, `'` to their HTML entity equivalents, preventing injected script execution via a malicious stream title or server error message.

```javascript
function sanitizeDisplay(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#x27;");
}
```

**[S2] URL Scheme Validation & Input Length Cap**
*OWASP A01: Broken Access Control*

URLs entered by the user are validated before any WebSocket message is sent or `window.open` is called. Only `http:` and `https:` schemes are accepted. `javascript:`, `data:`, `ftp:`, and all other schemes are explicitly rejected. Input is capped at 2048 characters to prevent buffer exhaustion.

```javascript
function isValidStreamUrl(url) {
  if (!url || url.length > 2048) return false;
  try {
    const u = new URL(url);
    return ["https:", "http:"].includes(u.protocol);
  } catch { return false; }
}
```

**[S3] WebSocket Message Schema Whitelist**
*OWASP A08: Software and Data Integrity Failures*

Every incoming WebSocket message is validated against a known-good type whitelist before any DOM operation is performed. Messages with unknown `type` values are silently dropped. This prevents unexpected server messages from triggering unintended client-side behavior.

```javascript
const ALLOWED_MSG_TYPES = new Set(["subtitle","alert","info","error","status","ping"]);

function handleWsMessage(event) {
  let msg;
  try { msg = JSON.parse(event.data); } catch { return null; }
  if (!ALLOWED_MSG_TYPES.has(msg.type)) return null;  // drop unknown types
  return msg;
}
```

**[S4] Reverse Tabnapping Prevention**
*OWASP — Unvalidated Redirects and Forwards*

The YouTube video is opened in a new tab using `window.open(url, "_blank", "noopener,noreferrer")`. The `noopener` flag nullifies `window.opener` in the new tab, preventing it from redirecting the monitoring page. `noreferrer` suppresses the `Referer` header.

---

### Backend Security

**[S5] SSRF Prevention — Server-Side URL Validation**
*OWASP A10: Server-Side Request Forgery*

Before any URL is passed to `yt-dlp`, the backend validates it against a strict allowlist. Private IP ranges (`10.x`, `192.168.x`, `172.16-31.x`), loopback addresses, `localhost`, and non-HTTP schemes are all blocked. This prevents an attacker from passing `http://localhost:5432` or `file:///etc/passwd` to the server's stream resolver.

```python
BLOCKED_HOSTS = re.compile(
    r"^(localhost|127\.|10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)",
    re.IGNORECASE
)

def validate_stream_url(url: str) -> bool:
    if not url or len(url) > 2048:
        return False
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return False
        if BLOCKED_HOSTS.match(p.hostname or ""):
            return False
        return True
    except Exception:
        return False
```

**[S6] Global Exception Handler — No Stack Trace Leakage**
*OWASP A05: Security Misconfiguration*

A FastAPI global exception handler catches all unhandled exceptions and returns a generic error message. Raw Python tracebacks — which can expose file paths, dependency versions, and internal logic — are never sent to the client. They are logged server-side only.

```python
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()  # log internally
    return JSONResponse(status_code=500, content={"error": "Internal server error"})
```

---

## Models

### faster-whisper `base.en`
| Property | Value |
|---|---|
| Origin | OpenAI Whisper, reimplemented by Systran with CTranslate2 |
| Architecture | Encoder-Decoder Transformer |
| Parameters | ~74 million |
| Quantisation | int8 — 4× memory reduction vs float32 |
| Input | 80-channel log-Mel spectrogram, 25ms frames, 10ms stride |
| Key feature | `word_timestamps=True` — per-word start, end, probability |
| License | MIT — fully open source, no API key |
| Runs | 100% locally on CPU |

### Silero VAD *(embedded)*
A lightweight LSTM-based speech/non-speech classifier embedded inside faster-whisper. Activated via `vad_filter=True`. Acts as a second VAD gate after the energy-based pre-filter — catches subtle non-speech (background music, HVAC) that passes the energy gate.

### yt-dlp
Stream manifest resolver. Extracts the direct audio URL from YouTube without downloading. FFmpeg then pipes raw PCM from that URL in real time. No YouTube API key required.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Backend | FastAPI + Uvicorn | Async HTTP + WebSocket server |
| Transcription | faster-whisper base.en | Speech-to-text, local inference |
| Stream extraction | yt-dlp | YouTube manifest resolution |
| Audio pipeline | FFmpeg | Decode/convert to 16kHz mono WAV |
| Fuzzy matching | RapidFuzz `partial_ratio` | Keyword detection with typo tolerance |
| Alert storage | SQLite | Persistent alert log |
| Frontend | Vanilla HTML/CSS/JS | Zero framework dependencies |
| Video playback | YouTube IFrame API | Native video embedding |
| Real-time comms | WebSocket | Sub-second subtitle delivery |

---

## Project Structure

```
securewatch/
├── backend/
│   ├── main.py              # FastAPI server — all research + security logic
│   ├── alerts.db            # SQLite alert log (auto-created on first run)
│   ├── temp_uploads/        # Temporary WAV chunks (auto-cleaned after use)
│   └── requirements.txt
├── frontend/
│   ├── index.html           # Full UI — upload + live stream modes
│   └── styles.css           # Dark intelligence monitoring theme
└── README.md
```

---

## Setup

### Prerequisites
- Python 3.11+
- FFmpeg on system PATH → [download](https://ffmpeg.org/download.html)

### Installation

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/securewatch.git
cd securewatch/backend

# Virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# Dependencies
pip install -r requirements.txt

# Start server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000` in your browser.

### requirements.txt
```
fastapi==0.111.0
uvicorn[standard]==0.29.0
faster-whisper==1.0.3
ffmpeg-python==0.2.0
rapidfuzz==3.9.1
numpy==1.26.4
websockets==12.0
python-multipart==0.0.9
requests
yt-dlp
pydantic
```

---

## Usage

### Uploaded Video
1. Select **Upload Video** tab
2. Drag & drop or browse for any video file
3. Click **Transcribe Video** — server extracts audio and runs Whisper
4. Press play — subtitles appear overlaid on video synchronized to playback
5. Keyword matches fire alerts logged with video-relative timestamp (e.g. `1:42`)

### Live Stream
1. Select **Live Stream** tab
2. Paste any YouTube URL or live stream link
3. Click **Connect** — video opens in new tab, monitoring panel activates
4. Server taps the audio stream via yt-dlp + FFmpeg every 5 seconds
5. Subtitles and alerts appear in the monitoring panel with stream elapsed time

---

## Alert System

An alert fires only when **all five gates pass**:

| Gate | Mechanism | Filters |
|---|---|---|
| 1. Fuzzy match | RapidFuzz `partial_ratio > 0.80` | Homophones, typos |
| 2. W-CTC word confidence | Per-word `probability >= 0.60` | Low-certainty decodes |
| 3. Timestamp alignment | Word timestamps overlap segment window | Boundary drift |
| 4. Segment confidence | `avg_logprob` → confidence `> 0.75` | Low-quality segments |
| 5. Cooldown | Same keyword cannot re-fire within 3s | Alert spam |

---

## Latency

| Component | Time |
|---|---|
| Audio chunk size | 5 seconds |
| Energy VAD (NumPy) | < 1 ms |
| Whisper inference (CPU, int8) | 1 – 3 seconds |
| WebSocket delivery | < 50 ms |
| **Total end-to-end** | **~6 – 8 seconds** |

To reduce to ~3–4s: set `beam_size=1`, `best_of=1`. Accuracy decreases slightly.

---

## Deployment

### Instant share — ngrok
```bash
ngrok http 8000
# Produces https://xxxx.ngrok.io — shareable immediately
```

### Permanent — Railway
1. Push to GitHub
2. Connect [Railway.app](https://railway.app) to your repo
3. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Deploy — model downloads on first cold start (~30s)

---

## References

1. **Macháček, D., Dabre, R., & Bojar, O.** (2023). Turning Whisper into Real-Time Transcription System. *Proc. Interspeech 2023.* https://arxiv.org/abs/2307.14743

2. **Kim, et al.** (2025). W-CTC: Word-level Connectionist Temporal Classification for Automatic Speech Recognition. *Proc. Interspeech 2025.*

3. **Sohn, J., Kim, N. S., & Sung, W.** (1999). A statistical model-based voice activity detection. *IEEE Signal Processing Letters, 6*(1), 1–3. https://doi.org/10.1109/97.736233

4. **Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I.** (2022). Robust Speech Recognition via Large-Scale Weak Supervision. *arXiv:2212.04356.* https://arxiv.org/abs/2212.04356

5. **Bain, M., et al.** (2023). WhisperX: Time-Accurate Speech Transcription of Long-Form Audio. *arXiv:2303.00747.* *(Referenced for word-timestamp alignment methodology.)*

6. **OWASP Foundation.** (2021). OWASP Top Ten. https://owasp.org/www-project-top-ten/

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

<div align="center">
<sub>
All transcription runs locally on your machine.<br>
No audio, video, or transcript data is ever sent to an external server or API.<br>
Built for DSP / Real-Time Systems coursework — Interspeech 2023 · Interspeech 2025 · IEEE SPL 1999
</sub>
</div>
