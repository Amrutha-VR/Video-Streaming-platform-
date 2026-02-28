from typing import Generator, Optional

import numpy as np


def transcribe_chunks(
    chunks: Generator[bytes, None, None],
    model,
    chunk_size: list = None,
) -> Generator[tuple[bytes, Optional[str]], None, None]:
    if chunk_size is None:
        chunk_size = [0, 10, 5]
    encoder_chunk_look_back = 4
    decoder_chunk_look_back = 1
    chunk_stride = chunk_size[1] * 960
    cache = {}

    for chunk in chunks:
        speech = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        total_chunk_num = max(1, int((len(speech) - 1) / chunk_stride + 1))
        transcript_parts = []

        for i in range(total_chunk_num):
            start = i * chunk_stride
            end = min((i + 1) * chunk_stride, len(speech))
            speech_chunk = speech[start:end]
            if len(speech_chunk) == 0:
                continue
            is_final = i == total_chunk_num - 1
            try:
                res = model.generate(
                    input=speech_chunk,
                    cache=cache,
                    is_final=is_final,
                    chunk_size=chunk_size,
                    encoder_chunk_look_back=encoder_chunk_look_back,
                    decoder_chunk_look_back=decoder_chunk_look_back,
                )
                text = None
                if res and len(res) > 0:
                    item = res[0]
                    text = (item.get("text", "") if isinstance(item, dict) else str(item)).strip()
                if text:
                    transcript_parts.append(text)
                    print(f"[ASR] {text}")
            except Exception as e:
                print(f"[ASR error] {e}")

        transcript = " ".join(transcript_parts) if transcript_parts else None
        yield chunk, transcript
