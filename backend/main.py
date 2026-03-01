import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import ffmpeg
from services.audio_extractor import extract_audio_chunks
from services.asr_service import transcribe_chunks
from services.asr_streaming import StreamingASR


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    def _load():
        # Load local English-only Faster-Whisper model (CPU, int8)
        return StreamingASR()

    app.state.asr_model = await asyncio.to_thread(_load)
    yield
    app.state.asr_model = None


app = FastAPI(title="Video Monitoring System API", version="1.0.0", redoc_url=None, docs_url=None, lifespan=lifespan)

UPLOAD_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Video Monitoring System</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; max-width: 480px; margin: 4rem auto; padding: 2rem; }
    h1 { font-size: 1.5rem; margin-bottom: 1.5rem; }
    input[type="file"] { display: block; margin-bottom: 1rem; width: 100%; }
    button { background: #0d6efd; color: white; border: none; padding: 0.6rem 1.5rem; font-size: 1rem; cursor: pointer; border-radius: 6px; }
    button:hover { background: #0b5ed7; }
    button:disabled { background: #6c757d; cursor: not-allowed; }
    #status { margin-top: 1rem; font-size: 0.9rem; }
  </style>
</head>
<body>
  <h1>Upload Video</h1>
  <form id="form">
    <input type="file" name="video" accept="video/*" required>
    <button type="submit" id="btn">Execute</button>
  </form>
  <div id="status"></div>
  <script>
    document.getElementById("form").onsubmit = async (e) => {
      e.preventDefault();
      const btn = document.getElementById("btn");
      const status = document.getElementById("status");
      btn.disabled = true;
      status.textContent = "Processing...";
      const form = new FormData(e.target);
      try {
        const res = await fetch("/upload-video", { method: "POST", body: form });
        status.textContent = res.ok ? "Done. Audio stream received." : "Error: " + res.status;
      } catch (err) {
        status.textContent = "Error: " + err.message;
      }
      btn.disabled = false;
    };
  </script>
</body>
</html>
"""


@app.get("/docs", include_in_schema=False)
async def docs():
    return HTMLResponse(UPLOAD_HTML)


@app.post("/upload-video")
async def upload_video(request: Request, video: UploadFile):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            content = await video.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        model = request.app.state.asr_model
        wav_tmp = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wtmp:
                wav_tmp = Path(wtmp.name)
            ffmpeg.input(str(tmp_path)).output(str(wav_tmp), format='wav', ac=1, ar=16000).run(quiet=True, overwrite_output=True)
            text = model.transcribe_chunk(str(wav_tmp)) or ""
            return {"text": text}
        finally:
            if wav_tmp and wav_tmp.exists():
                wav_tmp.unlink(missing_ok=True)
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    model = websocket.app.state.asr_model
    try:
        while True:
            data = await websocket.receive_bytes()
            if not data or len(data) == 0:
                continue
            text = model.transcribe_pcm_bytes(data)
            if text:
                await websocket.send_json({"text": text})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@app.websocket("/ws/video-stream")
async def websocket_video_stream(websocket: WebSocket):
  await websocket.accept()
  model = websocket.app.state.asr_model
  try:
    while True:
      chunk = await websocket.receive_bytes()
      if not chunk or len(chunk) == 0:
        continue
      tmp_path = None
      try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as vf:
          vf.write(chunk)
          tmp_path = Path(vf.name)
        for pcm in extract_audio_chunks(tmp_path):
          text = model.transcribe_pcm_bytes(pcm)
          if text:
            await websocket.send_json({"text": text})
      finally:
        if tmp_path and tmp_path.exists():
          try:
            tmp_path.unlink(missing_ok=True)
          except Exception:
            pass
  except WebSocketDisconnect:
    pass
  except Exception:
    pass

