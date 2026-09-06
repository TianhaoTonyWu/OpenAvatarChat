"""Dedicated runtime latency tracer for OpenAvatarChat.

Writes two files under logs/:
  latency.jsonl  — one JSON object per event, easy to parse later
  latency.log    — human-readable EVENT / TURN lines

Call latency.setup() once after loguru is configured.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from engine_utils.directory_info import DirectoryInfo


def _pct(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def _fmt_ms(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.0f}ms"


class _Turn:
    __slots__ = (
        "turn_id", "kind", "session_id", "t0", "marks", "fields",
        "duplug_infer_ms", "summarized",
    )

    def __init__(self, turn_id: int, kind: str, session_id: str):
        self.turn_id = turn_id
        self.kind = kind
        self.session_id = session_id
        self.t0 = time.perf_counter()
        self.marks: Dict[str, float] = {}
        self.fields: Dict[str, Any] = {}
        self.duplug_infer_ms: List[float] = []
        self.summarized = False

    def delta_ms(self, later: str, earlier: str) -> Optional[float]:
        if later not in self.marks or earlier not in self.marks:
            return None
        return (self.marks[later] - self.marks[earlier]) * 1000.0

    def since_ms(self, key: str) -> Optional[float]:
        if key not in self.marks:
            return None
        return (self.marks[key] - self.t0) * 1000.0


class LatencyTracer:
    def __init__(self):
        self._lock = threading.Lock()
        self._ready = False
        self._jsonl_fp = None
        self._human_fp = None
        self._turns: Dict[str, _Turn] = {}
        self._turn_seq = defaultdict(int)
        self._infer_seq = defaultdict(int)

    def setup(self, log_dir: Optional[str] = None) -> None:
        if log_dir is None:
            log_dir = os.path.join(DirectoryInfo.get_project_dir(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        jsonl_path = os.path.join(log_dir, "latency.jsonl")
        human_path = os.path.join(log_dir, "latency.log")
        with self._lock:
            if self._ready:
                return
            self._jsonl_fp = open(jsonl_path, "a", encoding="utf-8")
            self._human_fp = open(human_path, "a", encoding="utf-8")
            self._ready = True
        self._write_human(
            f"===== latency tracer started pid={os.getpid()} "
            f"jsonl={jsonl_path} ====="
        )
        logger.info(f"[LATENCY] tracer ready: {human_path}")

    def begin_turn(self, session_id: str, kind: str, **fields: Any) -> int:
        session_id = session_id or "-"
        with self._lock:
            prev = self._turns.get(session_id)
            if prev is not None and not prev.summarized:
                self._summarize_turn_unlocked(prev, reason="superseded")
            self._turn_seq[session_id] += 1
            turn_id = self._turn_seq[session_id]
            turn = _Turn(turn_id, kind, session_id)
            turn.fields.update(fields)
            self._turns[session_id] = turn
        self.mark("turn", "begin", session_id=session_id, kind=kind, **fields)
        return turn_id

    def mark(self, module: str, event: str, session_id: str = "", **fields: Any) -> None:
        now = time.perf_counter()
        session_id = session_id or "-"
        key = f"{module}.{event}"
        rec: Dict[str, Any] = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "session_id": session_id,
            "module": module,
            "event": event,
        }
        rec.update({k: v for k, v in fields.items() if v is not None})
        with self._lock:
            turn = self._turns.get(session_id)
            if turn is not None:
                rec["turn_id"] = turn.turn_id
                rec["turn_kind"] = turn.kind
                rec["since_turn_ms"] = round((now - turn.t0) * 1000.0, 1)
                if key not in turn.marks:
                    turn.marks[key] = now
                turn.fields.update({k: v for k, v in fields.items() if v is not None})
            self._write_jsonl_unlocked(rec)
        extra = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
        since = rec.get("since_turn_ms")
        since_s = f" +{since:.0f}ms" if since is not None else ""
        turn_s = f" turn={rec.get('turn_id')}" if rec.get("turn_id") else ""
        self._write_human(
            f"EVENT {module}.{event}{turn_s}{since_s} sid={session_id} {extra}".rstrip()
        )
        if event in (
            "first_token", "first_chunk", "first_speaking_frame",
            "text_submitted", "matched", "speech_end",
        ):
            logger.info(
                f"[LATENCY] {module}.{event}{turn_s}{since_s} sid={session_id} {extra}".rstrip()
            )

    def sample(self, module: str, metric: str, value_ms: float,
               session_id: str = "", **fields: Any) -> None:
        """High-frequency metric. Always stored; human log only if slow."""
        session_id = session_id or "-"
        rec = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "session_id": session_id,
            "module": module,
            "event": metric,
            "ms": round(value_ms, 1),
        }
        rec.update({k: v for k, v in fields.items() if v is not None})
        with self._lock:
            turn = self._turns.get(session_id)
            if turn is not None:
                rec["turn_id"] = turn.turn_id
                if module == "duplug" and metric == "infer":
                    turn.duplug_infer_ms.append(value_ms)
            self._infer_seq[session_id] += 1
            seq = self._infer_seq[session_id]
            self._write_jsonl_unlocked(rec)
            slow = value_ms >= 80.0
            periodic = seq == 1 or seq % 50 == 0
        if slow or periodic:
            tag = "SLOW" if slow else "sample"
            self._write_human(
                f"{tag} {module}.{metric} {value_ms:.1f}ms sid={session_id} n={seq}"
            )

    def complete_turn(self, session_id: str, reason: str = "done") -> None:
        session_id = session_id or "-"
        with self._lock:
            turn = self._turns.get(session_id)
            if turn is None or turn.summarized:
                return
            self._summarize_turn_unlocked(turn, reason=reason)

    @contextmanager
    def span(self, module: str, event: str, session_id: str = "", **fields: Any):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            self.mark(module, event, session_id=session_id, ms=round(ms, 1), **fields)

    def _summarize_turn_unlocked(self, turn: _Turn, reason: str) -> None:
        if turn.summarized:
            return
        turn.summarized = True
        infers = turn.duplug_infer_ms
        summary = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "type": "turn_summary",
            "session_id": turn.session_id,
            "turn_id": turn.turn_id,
            "kind": turn.kind,
            "reason": reason,
            "elapsed_ms": round((time.perf_counter() - turn.t0) * 1000.0, 1),
            "marks_ms": {
                k: round((v - turn.t0) * 1000.0, 1) for k, v in turn.marks.items()
            },
            "deltas_ms": {
                "speech_start_to_end": _round(turn.delta_ms("vad.speech_end", "vad.speech_start")),
                "speech_end_to_asr_done": _round(turn.delta_ms("asr.finish", "vad.speech_end")),
                "speech_end_to_text": _round(turn.delta_ms("duplug.text_submitted", "vad.speech_end")),
                "text_to_llm_first": _round(turn.delta_ms("llm.first_token", "duplug.text_submitted")),
                "llm_ttft": _round(turn.delta_ms("llm.first_token", "llm.request_start")),
                "llm_total": _round(turn.delta_ms("llm.complete", "llm.request_start")),
                "llm_first_to_tts_first": _round(turn.delta_ms("tts.first_chunk", "llm.first_token")),
                "tts_ttfa": _round(turn.delta_ms("tts.first_chunk", "tts.request_start")),
                "tts_first_to_avatar_first": _round(
                    turn.delta_ms("musetalk.first_speaking_frame", "tts.first_chunk")
                ),
                "speech_end_to_tts_first": _round(turn.delta_ms("tts.first_chunk", "vad.speech_end")),
                "speech_end_to_avatar_first": _round(
                    turn.delta_ms("musetalk.first_speaking_frame", "vad.speech_end")
                ),
                "wake_speech_to_match": _round(
                    turn.delta_ms("wakeword.matched", "wakeword.speech_start")
                ),
                "wake_match_to_ack": _round(
                    turn.delta_ms("wakeword.ack_submitted", "wakeword.matched")
                ),
            },
            "duplug_infer": {
                "n": len(infers),
                "mean_ms": _round(sum(infers) / len(infers) if infers else None),
                "p50_ms": _round(_pct(infers, 50)),
                "p95_ms": _round(_pct(infers, 95)),
                "max_ms": _round(max(infers) if infers else None),
            },
            "fields": turn.fields,
        }
        self._write_jsonl_unlocked(summary)
        d = summary["deltas_ms"]
        lines = [
            f"TURN #{turn.turn_id} kind={turn.kind} reason={reason} "
            f"sid={turn.session_id} elapsed={summary['elapsed_ms']:.0f}ms",
            f"  VAD speech { _fmt_ms(d['speech_start_to_end']) }  |  "
            f"speech_end→ASR { _fmt_ms(d['speech_end_to_asr_done']) }  |  "
            f"speech_end→text { _fmt_ms(d['speech_end_to_text']) }",
            f"  LLM TTFT { _fmt_ms(d['llm_ttft']) }  total { _fmt_ms(d['llm_total']) }  |  "
            f"text→LLM first { _fmt_ms(d['text_to_llm_first']) }",
            f"  TTS first-audio { _fmt_ms(d['tts_ttfa']) }  |  "
            f"LLM first→TTS first { _fmt_ms(d['llm_first_to_tts_first']) }",
            f"  MuseTalk TTS→first-frame { _fmt_ms(d['tts_first_to_avatar_first']) }",
            f"  E2E speech_end→TTS first { _fmt_ms(d['speech_end_to_tts_first']) }  |  "
            f"speech_end→avatar first { _fmt_ms(d['speech_end_to_avatar_first']) }",
        ]
        if d["wake_speech_to_match"] is not None:
            lines.append(
                f"  Wake speech→match { _fmt_ms(d['wake_speech_to_match']) }  |  "
                f"match→ack { _fmt_ms(d['wake_match_to_ack']) }"
            )
        inf = summary["duplug_infer"]
        if inf["n"]:
            lines.append(
                f"  Duplug infer n={inf['n']} mean={_fmt_ms(inf['mean_ms'])} "
                f"p50={_fmt_ms(inf['p50_ms'])} p95={_fmt_ms(inf['p95_ms'])} "
                f"max={_fmt_ms(inf['max_ms'])}"
            )
        text = "\n".join(lines)
        self._write_human_unlocked(text)
        logger.info("[LATENCY]\n" + text)

    def _write_jsonl_unlocked(self, rec: Dict[str, Any]) -> None:
        if self._jsonl_fp is None:
            return
        try:
            self._jsonl_fp.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            self._jsonl_fp.flush()
        except Exception:
            pass

    def _write_human_unlocked(self, line: str) -> None:
        if self._human_fp is None:
            return
        stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        try:
            self._human_fp.write(f"{stamp} {line}\n")
            self._human_fp.flush()
        except Exception:
            pass

    def _write_human(self, line: str) -> None:
        with self._lock:
            self._write_human_unlocked(line)


def _round(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(value, 1)


latency = LatencyTracer()
