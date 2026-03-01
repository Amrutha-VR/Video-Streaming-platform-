from typing import Generator, Optional
import tempfile
import wave
from pathlib import Path


def transcribe_chunks(
    chunks: Generator[bytes, None, None],
    model,
    chunk_size: list = None,
) -> Generator[tuple[bytes, Optional[str]], None, None]:
    for chunk in chunks:
        transcript = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp_path = Path(tmp.name)
            
            with wave.open(str(tmp_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(chunk)

            text = model.transcribe_chunk(str(tmp_path))
            if text:
                transcript = text
                print(f"[ASR] {text}")
            else:
                print(f"[ASR] (empty)")
        except Exception as e:
            print(f"[ASR error] {e}")
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

        yield chunk, transcript
