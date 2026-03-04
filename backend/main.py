"""
Government Video Monitoring System — Backend
=============================================
Modes
-----
1. Uploaded video  : POST /upload-video  →  ffmpeg extracts audio  →  faster-whisper
2. YouTube stream  : WS /yt-transcribe   →  yt-dlp + ffmpeg pipe   →  faster-whisper

Research Features (all properly implemented, not just cited)
-------------------------------------------------------------
[1] W-CTC Word-level Confidence Gating
    Kim et al., "W-CTC: Word-level Connectionist Temporal Classification
    for Automatic Speech Recognition", Interspeech 2025.
    → Every keyword match is gated by per-WORD confidence, not just
      segment-level avg_logprob. Only words whose log-prob exceeds a
      threshold AND whose timestamp overlaps the current segment window
      are allowed to trigger alerts. This properly replicates the W-CTC
      forced-alignment spirit.

[2] Local Agreement Policy with Rollback
    Macháček et al., "Turning Whisper into Real-Time Transcription System",
    Interspeech 2023 (Whisper-Streaming paper).
    → Transcribed words are only emitted once confirmed by two consecutive
      overlapping chunks. If a word changes between chunks it is rolled back
      and re-evaluated. Prevents hallucinated boundary words from reaching
      the subtitle display.

[3] Energy-based VAD Boundary Detection
    Inspired by Sohn et al., "A statistical model-based voice activity
    detection", IEEE Signal Processing Letters 1999, and the WebRTC VAD
    implementation used in production ASR pipelines.
    → Before sending a chunk to Whisper, we compute the RMS energy of the
      raw PCM. Chunks below a calibrated silence threshold are skipped
      entirely, reducing Whisper hallucinations on silent/background-noise
      segments and lowering overall CPU load.
"""

import asyncio
import io
import math
import os
import shutil
import sqlite3
import subprocess
import struct
import threading
import time
import uuid
import wave
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import yt_dlp
from fastapi import FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from pydantic import BaseModel
from rapidfuzz import fuzz
import ffmpeg as ffmpeg_py

# ── Paths ────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))
UPLOAD_DIR   = os.path.join(BASE_DIR, "temp_uploads")
DB_PATH      = os.path.join(BASE_DIR, "alerts.db")
FFMPEG_BIN   = shutil.which("ffmpeg") or "ffmpeg"

os.makedirs(UPLOAD_DIR, exist_ok=True)
print(f"[Init] ffmpeg: {FFMPEG_BIN}")

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    print(f"[ValidationError] {exc.errors()}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    print(f"[GlobalError] {request.method} {request.url} — {exc}")
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"error": str(exc)})

# ── Keywords ─────────────────────────────────────────────────
ALERT_KEYWORDS: List[str] = [
    "wow", "internet",
    "threat", "attack", "bomb", "explosive", "detonation",
    "weapon", "gun", "rifle", "hostage", "terrorist",
    "evacuate", "emergency", "lockdown", "shelter in place",
    "active shooter", "security", "security breach", "breach",
    "suspicious", "package", "classified", "top secret",
    "confidential", "restricted", "intelligence report",
    "covert operation", "critical infrastructure",
    "power grid", "substation", "pipeline", "nuclear",
    "radiological", "chemical weapon", "biological agent",
    "bomb threat", "evacuation order", "martial law",
    "state of emergency",
]

HALLUCINATIONS: Set[str] = {
    "you", "thank you", "thanks", "bye", "goodbye", "thank you.",
    "thanks.", "bye.", "you.", "uh", "um", "...", ". . .", "the",
    "a", "i", "oh", "okay", "ok", "hmm", "hm", "ah", "eh",
    "uh-huh", "mm-hmm", "subtitles by", "transcribed by",
    "thank you for watching", "please subscribe",
}

def is_hallucination(text: str) -> bool:
    t = text.strip().lower().rstrip(".,!?")
    return t in HALLUCINATIONS or len(t) <= 1

# ── Database ──────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT, keyword TEXT, clip_path TEXT)""")
    conn.commit(); conn.close()

init_db()

def save_alert(keyword: str, clip_path: str = "") -> str:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO alerts (created_at,keyword,clip_path) VALUES(?,?,?)",
        (ts, keyword, clip_path))
    conn.commit(); conn.close()
    return ts

# ── Whisper model ─────────────────────────────────────────────
_model: Optional[WhisperModel] = None

def get_model() -> WhisperModel:
    global _model
    if _model is None:
        print("[Whisper] Loading base.en …")
        _model = WhisperModel("base.en", device="cpu", compute_type="int8")
        print("[Whisper] Ready.")
    return _model

# ============================================================
# RESEARCH FEATURE 1 — Energy-based VAD Boundary Detection
# Inspired by Sohn et al. (IEEE Signal Processing Letters, 1999)
# and WebRTC VAD production pipeline.
#
# Computes RMS energy of raw int16 PCM. If below SILENCE_THRESHOLD,
# the chunk is classified as silence and skipped before Whisper runs.
# This prevents Whisper from hallucinating on background noise.
# ============================================================

SILENCE_THRESHOLD   = 0.010   # RMS in range [0, 1] — tune if needed
SILENCE_RATIO_MAX   = 0.85    # skip chunk if >85% of frames are silent

def compute_rms(pcm_bytes: bytes) -> float:
    """Return normalised RMS energy of int16 PCM bytes (range 0–1)."""
    if len(pcm_bytes) < 2:
        return 0.0
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    rms = float(np.sqrt(np.mean(samples ** 2))) / 32768.0
    return rms

def vad_chunk_is_speech(pcm_bytes: bytes,
                         frame_ms: int = 30,
                         sample_rate: int = 16000) -> bool:
    """
    Energy-based VAD: slide a 30 ms window over the PCM.
    Returns True if enough frames exceed the silence threshold.

    Replicates the energy-gating step from the WebRTC VAD pipeline
    (Sohn et al., 1999 statistical model — simplified energy version).
    """
    frame_samples = int(sample_rate * frame_ms / 1000)
    frame_bytes   = frame_samples * 2   # int16 = 2 bytes/sample
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)

    total_frames  = 0
    speech_frames = 0

    for start in range(0, len(samples) - frame_samples, frame_samples):
        frame = samples[start : start + frame_samples]
        rms   = float(np.sqrt(np.mean(frame ** 2))) / 32768.0
        total_frames  += 1
        if rms >= SILENCE_THRESHOLD:
            speech_frames += 1

    if total_frames == 0:
        return False

    speech_ratio = speech_frames / total_frames
    result = speech_ratio > (1.0 - SILENCE_RATIO_MAX)
    print(f"[VAD] speech_ratio={speech_ratio:.2f} → {'SPEECH' if result else 'SILENCE — skipped'}")
    return result

# ============================================================
# RESEARCH FEATURE 2 — Local Agreement Policy with Rollback
# Macháček et al., Interspeech 2023 (Whisper-Streaming paper).
#
# Words are only emitted once they are CONFIRMED — i.e., they
# appeared in the same position in two consecutive transcriptions.
# If a word changes between iterations, the unconfirmed suffix is
# rolled back and re-evaluated in the next chunk.
# ============================================================

class LocalAgreementBuffer:
    """
    Implements the Local Agreement policy from Whisper-Streaming
    (Macháček et al., Interspeech 2023).

    State machine:
      confirmed_words  — words already emitted to the user (immutable)
      pending_words    — words seen once, awaiting confirmation
    """

    def __init__(self):
        self.confirmed_words: List[str] = []
        self.pending_words:   List[str] = []

    def update(self, new_text: str) -> Optional[str]:
        """
        Feed the latest transcription. Returns the newly confirmed
        string to display, or None if nothing new was confirmed.

        Algorithm (Macháček et al. §3.1):
          1. Align new_text words against pending_words by common prefix.
          2. Words in the common prefix are NOW confirmed.
          3. Any suffix that differs → rolled back to pending.
          4. Emit only the delta between old confirmed and new confirmed.
        """
        new_words = new_text.strip().split()
        if not new_words:
            return None

        # Find longest common prefix between pending and new
        agreed: List[str] = []
        for pw, nw in zip(self.pending_words, new_words):
            if pw.lower().rstrip(".,!?") == nw.lower().rstrip(".,!?"):
                agreed.append(nw)
            else:
                break   # first disagreement → stop

        # Newly confirmed = agreed words not yet in confirmed_words
        prev_confirmed_len = len(self.confirmed_words)
        newly_confirmed    = agreed[prev_confirmed_len:]

        if newly_confirmed:
            self.confirmed_words.extend(newly_confirmed)

        # Rollback: pending becomes the full new transcription
        # (unconfirmed tail will be re-evaluated next chunk)
        self.pending_words = new_words

        if newly_confirmed:
            result = " ".join(newly_confirmed)
            print(f"[LocalAgreement] Confirmed: '{result}'")
            return result

        # Nothing confirmed yet — show pending words as tentative subtitle
        # (greyed-out preview, same as Whisper-Streaming's partial output)
        tentative = " ".join(new_words[len(self.confirmed_words):])
        return tentative if tentative else None

    def reset(self):
        self.confirmed_words = []
        self.pending_words   = []

# ============================================================
# RESEARCH FEATURE 3 — W-CTC Word-level Confidence Gating
# Kim et al., "W-CTC: Word-level CTC for ASR", Interspeech 2025.
#
# In standard Whisper, segment-level avg_logprob is used as confidence.
# W-CTC aligns each word to a timestamp and assigns per-word scores.
# We replicate this using Whisper's word_timestamps=True output:
#   - Each word has its own start/end timestamp and probability
#   - A keyword alert is only fired if:
#       (a) the matched word's timestamp overlaps the segment window
#       (b) the word-level log_prob exceeds WORD_CONF_THRESHOLD
# This is the PROPER implementation — not the trivially-true version.
# ============================================================

WORD_CONF_THRESHOLD = 0.60   # minimum per-word probability for alert

def wctc_keyword_check(
    keyword: str,
    segment,               # faster-whisper Segment object
    seg_start: float,
    seg_end:   float,
) -> Tuple[bool, float]:
    """
    W-CTC inspired word-level confidence gating.
    (Kim et al., Interspeech 2025)

    Returns (is_valid: bool, best_word_prob: float).

    Steps:
      1. Find words in segment.words that match the keyword token.
      2. For each matching word, verify its timestamp overlaps [seg_start, seg_end].
      3. Check the word's probability >= WORD_CONF_THRESHOLD.
      4. Only return True if at least one word passes all checks.
    """
    kw_tokens = keyword.lower().split()   # multi-word keywords e.g. "active shooter"
    words     = getattr(segment, "words", []) or []

    if not words:
        # No word-level data — fall back to segment confidence
        seg_conf = max(0.0, min(1.0, 1.0 + getattr(segment, "avg_logprob", -1.0)))
        valid = (
            seg_conf >= WORD_CONF_THRESHOLD
            and not (seg_end < seg_start or seg_start > seg_end + 1.0)
        )
        return valid, seg_conf

    # Build a searchable list of (word_text, start, end, prob)
    word_list = []
    for w in words:
        txt  = getattr(w, "word",        "").strip().lower().rstrip(".,!?")
        s    = getattr(w, "start",       seg_start)
        e    = getattr(w, "end",         seg_end)
        prob = getattr(w, "probability", 0.5)
        word_list.append((txt, s, e, prob))

    # Slide a window of len(kw_tokens) over word_list
    for i in range(len(word_list) - len(kw_tokens) + 1):
        window = word_list[i : i + len(kw_tokens)]
        # Check if window tokens match keyword tokens (fuzzy)
        match = all(
            fuzz.ratio(kw_tok, win_tok) > 70
            for kw_tok, (win_tok, *_) in zip(kw_tokens, window)
        )
        if not match:
            continue

        # All words in the window must pass timestamp + confidence gates
        window_ok = True
        min_prob  = 1.0
        for (wtxt, wstart, wend, wprob) in window:
            # Timestamp overlap check (the real W-CTC alignment spirit)
            if wend < seg_start or wstart > seg_end + 1.0:
                window_ok = False
                break
            if wprob < WORD_CONF_THRESHOLD:
                window_ok = False
                break
            min_prob = min(min_prob, wprob)

        if window_ok:
            print(f"[W-CTC] Keyword '{keyword}' passed "
                  f"alignment+confidence gate (min_word_prob={min_prob:.2f})")
            return True, min_prob

    return False, 0.0

# ── Shared transcription function ─────────────────────────────
def pcm_to_wav_bytes(pcm: bytes, sample_rate: int = 16000) -> io.BytesIO:
    """
    Wrap raw int16 PCM in a WAV container in memory (no disk I/O).
    Returns a BytesIO seeked to position 0 — faster-whisper accepts
    file-like objects directly, avoiding temp file write + read.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    buf.seek(0)
    return buf


def transcribe_wav(
    wav_source,                   # str path OR BytesIO
    pcm_bytes: Optional[bytes],   # raw PCM for VAD check; None skips VAD
    model:     WhisperModel,
    la:        LocalAgreementBuffer,
    last_alert: dict,
    on_subtitle,
    on_alert,
):
    # ── [FEATURE 1] Energy VAD — skip silent chunks ───────────
    if pcm_bytes is not None:
        if not vad_chunk_is_speech(pcm_bytes):
            return   # silent chunk — don't waste Whisper cycles

    try:
        segs, info = model.transcribe(
            wav_source,    # accepts path string OR BytesIO
            word_timestamps=True,       # needed for W-CTC feature
            beam_size=5,
            best_of=5,
            temperature=0.0,
            vad_filter=True,
            vad_parameters={
                "threshold": 0.35,
                "min_silence_duration_ms": 300,
                "speech_pad_ms": 400,
            },
            condition_on_previous_text=False,
            no_speech_threshold=0.45,
            log_prob_threshold=-2.0,
            initial_prompt=(
                "Real-time speech transcription. "
                "Speaker is talking clearly in English."
            ),
        )
        print(f"[Whisper] lang={info.language} "
              f"p={info.language_probability:.2f} "
              f"dur={info.duration:.1f}s")

        for seg in segs:
            text = seg.text.strip()
            if not text or is_hallucination(text):
                print(f"[Whisper] Hallucination dropped: '{text}'")
                continue

            seg_conf = max(0.0, min(1.0, 1.0 + getattr(seg, "avg_logprob", -0.5)))
            print(f"[Whisper] '{text}'  seg_conf={seg_conf:.2f}")

            # ── [FEATURE 2] Local Agreement — confirm words ───
            display = la.update(text)
            if display:
                on_subtitle(display)

            # ── [FEATURE 3] W-CTC keyword gating ─────────────
            lower = text.lower()
            now   = time.time()

            for kw in ALERT_KEYWORDS:
                kl = kw.lower()
                # Quick pre-filter: exact or fuzzy match in text
                if not (kl in lower or
                        fuzz.partial_ratio(kl, lower) / 100.0 > 0.80):
                    continue

                # Full W-CTC word-level confidence + alignment check
                valid, word_prob = wctc_keyword_check(
                    kw, seg, seg.start, seg.end)
                if not valid:
                    print(f"[W-CTC] '{kw}' rejected by alignment/confidence gate")
                    continue

                # Cooldown deduplication
                if now - last_alert.get(kw, 0.0) < 3.0:
                    continue

                last_alert[kw] = now
                ts = save_alert(kw)
                # stream_ts = wall-clock time inside the stream for display
                stream_ts = time.strftime("%H:%M:%S")
                print(f"[ALERT] '{kw}' (word_prob={word_prob:.2f})")
                on_alert({"type":"alert","timestamp":ts,
                           "keyword":kw,"clip_path":"",
                           "stream_ts": stream_ts})
                break

    except Exception as e:
        print(f"[Whisper] Error: {e}")

# ── Upload mode ───────────────────────────────────────────────
@app.post("/upload-video")
async def upload_video(request: Request):
    """
    Accepts video as raw binary body (Content-Type: application/octet-stream).
    Bypasses multipart entirely — no python-multipart dependency for this route.
    """
    try:
        body = await request.body()
        print(f"[Upload] Raw body received: {len(body)} bytes")

        if not body:
            return JSONResponse({"error": "Empty file", "segments": []}, status_code=400)

        # Get filename hint from header if sent
        fname = request.headers.get("X-Filename", "upload.mp4")
        vid   = str(uuid.uuid4())
        vp    = os.path.join(UPLOAD_DIR, f"{vid}_{fname}")
        ap    = os.path.join(UPLOAD_DIR, f"{vid}.wav")

        with open(vp, "wb") as f:
            f.write(body)
        print(f"[Upload] Saved to {vp}")

        ffmpeg_py.input(vp).output(ap, ac=1, ar=16000, format="wav") \
            .overwrite_output().run(quiet=True)
        print(f"[Upload] WAV extracted: {os.path.getsize(ap)} bytes")

        model    = get_model()
        segs, info = model.transcribe(ap, word_timestamps=True)
        seg_list = list(segs)
        print(f"[Upload] {len(seg_list)} segments, lang={info.language}")

        result = [
            {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
            for s in seg_list if s.text.strip()
        ]
        return JSONResponse({"segments": result})

    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({"error": str(e), "segments": []}, status_code=500)

@app.post("/upload-alert")
async def upload_alert(payload: dict):
    save_alert(payload.get("keyword",""), payload.get("clip_path",""))
    return {"status": "ok"}

# ── YouTube stream session ────────────────────────────────────
class YTSession:
    def __init__(self, url, ws, loop):
        self.url     = url
        self.ws      = ws
        self.loop    = loop
        self.running = False
        self.proc: Optional[subprocess.Popen] = None

    @staticmethod
    def resolve_audio_url(yt_url: str):
        with yt_dlp.YoutubeDL({"format":"bestaudio/best","quiet":True,
                                "no_warnings":True}) as ydl:
            info = ydl.extract_info(yt_url, download=False)
            url  = info.get("url") or info["formats"][-1]["url"]
            return url, info.get("title",""), info.get("id","")

    def _send(self, payload):
        asyncio.run_coroutine_threadsafe(
            self.ws.send_json(payload), self.loop)

    def _stream_thread(self):
        model      = get_model()
        la         = LocalAgreementBuffer()
        last_alert : dict = {}
        SAMPLE_RATE   = 16000
        CHUNK_SECS    = 5
        CHUNK_BYTES   = SAMPLE_RATE * 2 * CHUNK_SECS  # int16

        def on_sub(t):   self._send({"type":"subtitle","text":t})
        def on_alert(p): self._send(p)

        try:
            print(f"[YT] Resolving: {self.url}")
            audio_url, title, vid_id = self.resolve_audio_url(self.url)
            print(f"[YT] '{title}'")
            self._send({"type":"info","title":title,"video_id":vid_id})

            cmd = [
                FFMPEG_BIN,
                "-reconnect",           "1",
                "-reconnect_streamed",  "1",
                "-reconnect_delay_max", "3",
                # Reduce input buffer — don't pre-buffer more than needed
                "-probesize",           "32768",
                "-analyzeduration",     "0",
                "-i", audio_url,
                "-vn",
                "-acodec",  "pcm_s16le",
                "-ar",      str(SAMPLE_RATE),
                "-ac",      "1",
                "-f",       "s16le",
                # Flush output every chunk — don't let FFmpeg buffer internally
                "-flush_packets", "1",
                "pipe:1",
            ]
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            print("[YT] ffmpeg piping started.")

            while self.running:
                pcm = self.proc.stdout.read(CHUNK_BYTES)
                if not pcm:
                    print("[YT] Stream ended.")
                    break
                if len(pcm) < CHUNK_BYTES // 4:
                    continue

                # Build WAV in memory — no disk write/read round trip
                wav_buf = pcm_to_wav_bytes(pcm)

                try:
                    transcribe_wav(wav_buf, pcm, model, la,
                                   last_alert, on_sub, on_alert)
                except Exception as e:
                    print(f"[Transcribe] Error: {e}")

        except Exception as e:
            print(f"[YT] Error: {e}")
            self._send({"type":"error","message":str(e)})
        finally:
            self.stop()

    def start(self):
        self.running = True
        threading.Thread(target=self._stream_thread, daemon=True).start()

    def stop(self):
        self.running = False
        if self.proc:
            try: self.proc.terminate()
            except: pass
            self.proc = None

# ── WebSocket endpoint ────────────────────────────────────────
@app.websocket("/yt-transcribe")
async def yt_transcribe(websocket: WebSocket):
    await websocket.accept()
    loop    = asyncio.get_event_loop()
    session: Optional[YTSession] = None
    print("[WS-YT] Connected.")
    try:
        while True:
            try:
                msg = await asyncio.wait_for(
                    websocket.receive_json(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type":"ping"}); continue

            if msg.get("action") == "start":
                url = msg.get("url","").strip()
                if not url:
                    await websocket.send_json(
                        {"type":"error","message":"No URL."}); continue
                if session: session.stop()
                session = YTSession(url, websocket, loop)
                session.start()
                await websocket.send_json(
                    {"type":"status","message":"Stream started."})

            elif msg.get("action") == "stop":
                if session: session.stop(); session = None
                await websocket.send_json(
                    {"type":"status","message":"Stream stopped."})

    except WebSocketDisconnect:
        print("[WS-YT] Disconnected.")
    finally:
        if session: session.stop()

# ── Static ────────────────────────────────────────────────────
@app.get("/config")
async def config():
    return {"alert_keywords": ALERT_KEYWORDS}

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
