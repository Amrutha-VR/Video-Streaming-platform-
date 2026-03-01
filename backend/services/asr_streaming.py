from faster_whisper import WhisperModel
import numpy as np
import wave
from pathlib import Path
from collections import deque
from typing import Optional, List
import time


class StreamingASR:
    def __init__(self, model_size: str = "base.en"):
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.sample_rate = 16000
        self.window_seconds = 4
        self.step_seconds = 1
        self.max_buffer_seconds = 5
        self.window_samples = self.sample_rate * self.window_seconds
        self.max_buffer_samples = self.sample_rate * self.max_buffer_seconds
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.committed_text = ""
        self.previous_partial = ""
        self._committed_words: List[str] = []
        self._word_first_seen: dict = {}
        self._commit_latencies = deque(maxlen=200)
        self._rtf_history = deque(maxlen=20)
        self._last_low_conf_candidate: Optional[str] = None

    def transcribe_pcm_bytes(self, pcm_bytes: bytes) -> Optional[str]:
        if not pcm_bytes or len(pcm_bytes) == 0:
            return None
        
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio = np.clip(audio, -1.0, 1.0)
        
        self.audio_buffer = np.concatenate((self.audio_buffer, audio))
        if len(self.audio_buffer) > self.max_buffer_samples:
            self.audio_buffer = self.audio_buffer[-self.max_buffer_samples:]
        
        if len(self.audio_buffer) < self.window_samples:
            return None
        
        window_audio = self.audio_buffer[-self.window_samples:]
        
        try:
            t0 = time.time()
            segments, _ = self.model.transcribe(
                window_audio,
                language="en",
                beam_size=1,
                temperature=0,
                vad_filter=False
            )
            inference_time = time.time() - t0
            rtf = inference_time / float(self.window_seconds)
            self._rtf_history.append(rtf)
            avg_rtf = float(sum(self._rtf_history) / len(self._rtf_history)) if len(self._rtf_history) else 0.0
            if avg_rtf > 1.0:
                self.step_seconds = min(self.step_seconds + 0.2, 2.0)
            elif avg_rtf < 0.5:
                self.step_seconds = max(self.step_seconds - 0.2, 0.5)
            self.step_samples = int(self.sample_rate * self.step_seconds)

            partial_text = " ".join(seg.text for seg in segments).strip()
            log_probs = [getattr(seg, "avg_logprob", None) for seg in segments]
            log_probs = [float(lp) for lp in log_probs if lp is not None]
            avg_logprob = float(np.mean(log_probs)) if log_probs else None
            confidence = float(np.exp(avg_logprob)) if avg_logprob is not None else 1.0
            confidence = max(0.0, min(confidence, 1.0))
        except Exception:
            return None
        
        if not partial_text:
            return None

        partial_words = partial_text.split()
        for w in partial_words:
            if w not in self._committed_words and w not in self._word_first_seen:
                self._word_first_seen[w] = time.time()

        if confidence < 0.85:
            if partial_text != self._last_low_conf_candidate:
                self._last_low_conf_candidate = partial_text
                self.previous_partial = partial_text
                return None
        self._last_low_conf_candidate = None

        prev_words = self.previous_partial.split()
        cur_words = partial_words
        i = 0
        m = min(len(prev_words), len(cur_words))
        while i < m and prev_words[i] == cur_words[i]:
            i += 1
        lcp_words = cur_words[:i]

        committed_len = len(self._committed_words)
        if len(lcp_words) > committed_len:
            new_words = lcp_words[committed_len:]
            if new_words:
                self._committed_words.extend(new_words)
                self.committed_text = " ".join(self._committed_words).strip()
                now = time.time()
                for w in new_words:
                    if w in self._word_first_seen:
                        latency = now - self._word_first_seen[w]
                        self._commit_latencies.append(latency)
                        del self._word_first_seen[w]
                self.previous_partial = partial_text
                return " ".join(new_words).strip()

        self.previous_partial = partial_text
        return None

    def transcribe_chunk(self, wav_path: str) -> Optional[str]:
        try:
            segments, _ = self.model.transcribe(
                str(wav_path),
                language="en",
                beam_size=1,
                temperature=0,
                vad_filter=False,
            )
            text = " ".join(seg.text for seg in segments).strip()
            return text if text else None
        except Exception:
            return None

