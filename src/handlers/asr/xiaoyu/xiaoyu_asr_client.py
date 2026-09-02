"""Xiaoyu realtime ASR — official WebSocket API.

Docs: /root/小语语音识别服务API文档.html

Evaluated call styles:
1. HTTP /asr/single (old Yunnan gateway) — not in this HTML, queued, high latency.
2. WS dump-all-then-EOF — doc warns 发送过快可能影响识别结果; only the start is decoded.
3. WS file replay with 80ms / 6144-byte pacing, continuous_decoding=False —
   OK for offline files, too slow to start until the whole utterance is buffered.
4. WS streaming while the user speaks, 80ms / 6144-byte binary PCM, EOF at
   utterance end, continuous_decoding=True — official realtime path.

This system uses (4). Wake-word one-shot uses paced (3).
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Optional
from urllib.parse import urlencode

import numpy as np
from loguru import logger

DEFAULT_WS_URL = "ws://speech.xiaoyuzhineng.com:12392/"
DEFAULT_API_KEY = "223707f61bf7752c880e"
DEFAULT_ASR_URL = DEFAULT_WS_URL

# Official: 每 80ms 发送 1024*6 字节 pcm_s16le @ 16kHz.
PCM_CHUNK_BYTES = 1024 * 6
PCM_CHUNK_INTERVAL = 0.08


def float_to_pcm16(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    audio = np.asarray(audio).astype(np.float32).squeeze()
    if audio.size == 0:
        return b""
    if sample_rate != 16000:
        try:
            import soxr
            audio = soxr.resample(audio, sample_rate, 16000)
        except Exception:
            pass
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.5:
        audio = audio / 32767.0
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def _parse(raw) -> tuple[str, str]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    data = json.loads(raw)
    action = str(data.get("action") or "")
    nbest = data.get("nbest") or []
    text = ""
    if nbest and isinstance(nbest[0], dict):
        text = str(nbest[0].get("result") or "").strip()
    return action, text


def _ws_url(asr_url: str, api_key: str, continuous_decoding: bool) -> str:
    params = urlencode({
        "apikey": api_key,
        "ts": str(int(time.time())),
        "lang": "zh",
        "continuous_decoding": "True" if continuous_decoding else "False",
    })
    base = asr_url if str(asr_url).startswith("ws") else DEFAULT_WS_URL
    if not base.endswith("/"):
        base += "/"
    return f"{base}?{params}"


class XiaoyuLiveSession:
    """One WebSocket per user utterance: feed PCM as it arrives, EOF when done."""

    def __init__(
        self,
        asr_url: str = DEFAULT_WS_URL,
        api_key: str = DEFAULT_API_KEY,
        continuous_decoding: bool = True,
    ):
        self.asr_url = asr_url
        self.api_key = api_key
        self.continuous_decoding = continuous_decoding
        self.partial_text = ""
        self.final_text = ""
        self._final_segments: list[str] = []
        self._q: queue.Queue = queue.Queue()
        self._done = threading.Event()
        self._eof_sent = False
        self._worker = threading.Thread(target=self._run, name="xiaoyu-live", daemon=True)
        self._worker.start()

    def feed(self, audio: np.ndarray, sample_rate: int = 16000):
        pcm = float_to_pcm16(audio, sample_rate)
        if pcm:
            self._q.put(("pcm", pcm))

    def finish(self, timeout: float = 8.0) -> str:
        self._q.put(("eof", None))
        self._done.wait(timeout=timeout)
        text = (self.final_text or self.partial_text or "").strip()
        if self._final_segments:
            joined = "".join(self._final_segments).strip()
            # Prefer the longer of joined segments vs last partial/final.
            if len(joined) > len(text):
                text = joined
        logger.info(f"XiaoyuLive speech_end text='{text}'")
        return text

    def close(self):
        try:
            self._q.put_nowait(("close", None))
        except queue.Full:
            pass

    def _remember_final(self, text: str):
        text = (text or "").strip()
        if not text:
            return
        if self._final_segments:
            prev = self._final_segments[-1]
            # Growing hypothesis for the same segment → replace.
            if text.startswith(prev) or prev.startswith(text):
                self._final_segments[-1] = text if len(text) >= len(prev) else prev
            # Duplicate → ignore.
            elif text == prev:
                pass
            else:
                self._final_segments.append(text)
        else:
            self._final_segments.append(text)
        self.final_text = "".join(self._final_segments)

    def _apply(self, raw, after_eof: bool = False) -> str:
        action, text = _parse(raw)
        if text:
            self.partial_text = text
            if action in ("final_result", "speech_end"):
                self._remember_final(text)
            elif after_eof and not self.final_text:
                self._remember_final(text)
        return action

    def _run(self):
        import websocket

        ws = None
        pending = bytearray()
        last_send = 0.0
        try:
            url = _ws_url(self.asr_url, self.api_key, self.continuous_decoding)
            ws = websocket.create_connection(url, timeout=8)
            ws.settimeout(0.02)
            logger.info("XiaoyuLive connected")
            while True:
                try:
                    kind, payload = self._q.get(timeout=0.02)
                except queue.Empty:
                    kind, payload = None, None
                if kind == "close":
                    break
                if kind == "pcm" and payload:
                    pending.extend(payload)
                if kind == "eof":
                    while True:
                        try:
                            extra_kind, extra_payload = self._q.get_nowait()
                        except queue.Empty:
                            break
                        if extra_kind == "pcm" and extra_payload:
                            pending.extend(extra_payload)
                    while pending:
                        n = min(len(pending), PCM_CHUNK_BYTES)
                        wait = PCM_CHUNK_INTERVAL - (time.monotonic() - last_send)
                        if last_send > 0 and wait > 0:
                            time.sleep(wait)
                        ws.send_binary(bytes(pending[:n]))
                        del pending[:n]
                        last_send = time.monotonic()
                        self._drain(ws, after_eof=False)
                    ws.send("EOF")
                    self._eof_sent = True
                    ws.settimeout(0.3)
                    deadline = time.time() + 2.5
                    while time.time() < deadline:
                        try:
                            raw = ws.recv()
                        except Exception:
                            continue
                        if self._apply(raw, after_eof=True) == "speech_end":
                            break
                    break
                now = time.monotonic()
                while len(pending) >= PCM_CHUNK_BYTES:
                    wait = PCM_CHUNK_INTERVAL - (now - last_send)
                    if last_send > 0 and wait > 0:
                        time.sleep(wait)
                    ws.send_binary(bytes(pending[:PCM_CHUNK_BYTES]))
                    del pending[:PCM_CHUNK_BYTES]
                    last_send = time.monotonic()
                    now = last_send
                    # Mid-utterance pauses may emit speech_end/final_result; keep
                    # the socket alive until we send EOF so the rest of a long
                    # sentence with pauses is still recognized.
                    self._drain(ws, after_eof=False)
                self._drain(ws, after_eof=False)
        except Exception as exc:
            logger.warning(f"XiaoyuLive failed: {exc}")
        finally:
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass
            self._done.set()

    def _drain(self, ws, after_eof: bool = False):
        while True:
            try:
                raw = ws.recv()
            except Exception:
                break
            action = self._apply(raw, after_eof=after_eof)
            # Only treat speech_end as session completion after we sent EOF.
            # Otherwise a pause inside a long utterance would kill the worker
            # and later audio would be dropped (leaving only a middle fragment).
            if after_eof and action == "speech_end":
                self._done.set()
                break


def recognize(
    audio: np.ndarray,
    sample_rate: int = 16000,
    session_id: Optional[str] = None,
    asr_url: str = DEFAULT_WS_URL,
    api_key: str = DEFAULT_API_KEY,
    timeout: float = 15.0,
) -> str:
    """Offline helper: paced 80ms chunks then EOF (official file-replay demo)."""
    import websocket

    pcm = float_to_pcm16(audio, sample_rate=sample_rate)
    if not pcm:
        return ""
    url = _ws_url(asr_url, api_key, continuous_decoding=True)
    ws = websocket.create_connection(url, timeout=8)
    try:
        for i in range(0, len(pcm), PCM_CHUNK_BYTES):
            ws.send_binary(pcm[i:i + PCM_CHUNK_BYTES])
            time.sleep(PCM_CHUNK_INTERVAL)
        ws.send("EOF")
        ws.settimeout(timeout)
        final = ""
        while True:
            raw = ws.recv()
            action, text = _parse(raw)
            if text:
                final = text
            if action == "speech_end":
                break
        logger.info(f"XiaoyuASR text='{final}'")
        return final
    finally:
        try:
            ws.close()
        except Exception:
            pass
