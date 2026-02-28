import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

from services.audio_extractor import extract_audio_chunks
from services.asr_service import transcribe_chunks


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    from funasr import AutoModel

    def _load():
        return AutoModel(model="paraformer-zh-streaming", device="cpu")

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
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    def generate():
        try:
            model = request.app.state.asr_model
            for chunk, _ in transcribe_chunks(extract_audio_chunks(tmp_path), model):
                yield chunk
        finally:
            tmp_path.unlink(missing_ok=True)

    return StreamingResponse(
        generate(),
        media_type="application/octet-stream",
    )
