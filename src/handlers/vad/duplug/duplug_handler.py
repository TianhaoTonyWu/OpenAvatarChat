"""
SoulX-Duplug turn-taking + cascade ASR.

Aligned with official dialogue-system:
  Duplug `nonidle` -> barge-in
  Duplug `speak`   -> take the turn and submit `text` (Paraformer/SenseVoice)
"""

from __future__ import annotations

import base64
import json
import queue
import re
import threading
import time
from typing import Any, Dict, Optional, cast

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from chat_engine.common.handler_base import (
    HandlerBase,
    HandlerBaseInfo,
    HandlerDataInfo,
    HandlerDetail,
)
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_engine_config_data import (
    ChatEngineConfigModel,
    HandlerBaseConfigModel,
)
from chat_engine.data_models.chat_signal import ChatSignal, SignalFilterRule
from chat_engine.data_models.chat_signal_type import ChatSignalSourceType, ChatSignalType
from chat_engine.data_models.internal.handler_definition_data import ChatDataConsumeMode
from chat_engine.data_models.runtime_data.data_bundle import (
    DataBundle,
    DataBundleDefinition,
    DataBundleEntry,
)
from engine_utils.latency_tracer import latency


BACKCHANNELS = {
    "嗯", "嗯嗯", "啊", "啊啊", "哦", "哦哦", "噢", "哎", "呃",
    "哼", "哼哼", "嗯哼", "嘿", "哈哈", "呵呵",
    "好", "好的", "对", "是", "是的",
    "ok", "okay", "yeah", "hmm", "mm", "uh",
}


class DuplugConfigModel(HandlerBaseConfigModel, BaseModel):
    server_url: str = Field(default="ws://127.0.0.1:8000/turn")
    chunk_samples: int = Field(default=2560)
    sample_rate: int = Field(default=16000)
    request_timeout: float = Field(default=5.0)
    min_turn_chars: int = Field(default=2)
    min_utterance_seconds: float = Field(default=0.35)
    interrupt_on_nonidle: bool = Field(default=True)
    min_interrupt_chars: int = Field(
        default=2,
        description="Barge-in only after live ASR has this many real chars (noise usually has none)",
    )
    interrupt_cooldown_seconds: float = Field(default=0.4)
    interrupt_grace_seconds: float = Field(
        default=0.4,
        description="Barge-in only if Duplug stays nonidle this long (filters noise spikes)",
    )
    min_interrupt_rms: float = Field(
        default=0.045,
        description="Barge-in only if the current mic chunk is near-field loud",
    )
    min_eou_rms: float = Field(
        default=0.04,
        description="Ignore Duplug speak if the utterance never reached near-field energy",
    )
    min_feed_rms: float = Field(
        default=0.03,
        description="Below this RMS (~-30 dBFS), treat as far-field/noise and send silence",
    )
    nearfield_lock_rms: float = Field(
        default=0.08,
        description="Lock onto a close talker once a chunk is at least this loud",
    )
    nearfield_lock_seconds: float = Field(
        default=2.5,
        description="Keep rejecting much quieter overlapping talkers after a near-field lock",
    )


class _DuplugWsClient:
    """Non-blocking Duplug client. Drops oldest audio if inference lags.

    State is applied in the worker via on_result as soon as the server replies
    (official dialogue-system interrupts on nonidle in the same process() call).
    """

    def __init__(
        self,
        server_url: str,
        timeout: float,
        on_result: Optional[Any] = None,
    ):
        self.server_url = server_url
        self.timeout = timeout
        self.on_result = on_result
        self._ws = None
        self._in_q: queue.Queue = queue.Queue(maxsize=48)
        self._lock = threading.Lock()
        self.latest: Dict[str, Any] = {"state": {"state": "idle"}}
        self._stop = threading.Event()
        self._infer_count = 0
        self._worker = threading.Thread(target=self._run, name="duplug-ws", daemon=True)
        self._worker.start()

    def close(self):
        self._stop.set()
        try:
            self._in_q.put_nowait(None)
        except queue.Full:
            pass
        with self._lock:
            self._close_unlocked()

    def submit(self, session_id: str, audio_chunk: np.ndarray):
        item = (session_id, np.asarray(audio_chunk, dtype=np.float32).copy())
        try:
            self._in_q.put_nowait(item)
        except queue.Full:
            try:
                self._in_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._in_q.put_nowait(item)
            except queue.Full:
                pass

    def _close_unlocked(self):
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def _connect_locked(self):
        import websocket

        self._close_unlocked()
        ws = websocket.create_connection(self.server_url, timeout=self.timeout)
        # Keepalive longer than one GPU-contended turn inference.
        ws.settimeout(max(self.timeout, 5.0))
        self._ws = ws
        logger.info(f"Duplug WS connected to {self.server_url}")

    def _infer(self, session_id: str, audio_chunk: np.ndarray) -> Optional[Dict[str, Any]]:
        payload = json.dumps({
            "type": "audio",
            "session_id": session_id,
            "audio": base64.b64encode(np.asarray(audio_chunk, dtype=np.float32).tobytes()).decode(),
        })
        t0 = time.perf_counter()
        raw = None
        with self._lock:
            try:
                if self._ws is None:
                    self._connect_locked()
                self._ws.send(payload)
                raw = self._ws.recv()
            except Exception as exc:
                logger.warning(f"Duplug WS error, reconnecting: {exc}")
                try:
                    self._connect_locked()
                    self._ws.send(payload)
                    raw = self._ws.recv()
                except Exception as retry_exc:
                    logger.warning(f"Duplug WS retry failed: {retry_exc}")
                    self._close_unlocked()
        if raw is None:
            latency.sample("duplug", "infer", (time.perf_counter() - t0) * 1000.0,
                           session_id=session_id, ok=False)
            return None
        try:
            data = json.loads(raw)
        except Exception:
            latency.sample("duplug", "infer", (time.perf_counter() - t0) * 1000.0,
                           session_id=session_id, ok=False)
            return None
        latency.sample("duplug", "infer", (time.perf_counter() - t0) * 1000.0,
                       session_id=session_id, ok=True)
        return data if isinstance(data, dict) else None

    def _run(self):
        while not self._stop.is_set():
            try:
                item = self._in_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            session_id, piece = item
            result = self._infer(session_id, piece)
            if not result:
                continue
            self.latest = result
            self._infer_count += 1
            st = ((result.get("state") or {}).get("state"))
            rms = float(np.sqrt(np.mean(np.square(piece)))) if piece.size else 0.0
            if (
                self._infer_count == 1
                or self._infer_count % 50 == 0
                or st not in ("idle", "blank", None)
                or rms >= 0.01
            ):
                logger.info(
                    f"Duplug WS ok infer#{self._infer_count} state={st} "
                    f"rms={rms:.4f} q={self._in_q.qsize()}"
                )
            if self.on_result is not None:
                try:
                    self.on_result(result, rms)
                except TypeError:
                    self.on_result(result)
                except Exception as exc:
                    logger.opt(exception=True).warning(f"Duplug on_result failed: {exc}")


class DuplugContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: DuplugConfigModel = DuplugConfigModel()
        self.shared_states = None
        self.client: Optional[_DuplugWsClient] = None
        self.audio_buffer = np.zeros((0,), dtype=np.float32)
        self.last_state: str = "idle"
        self.saw_nonidle: bool = False
        self.idle_after_speech: int = 0
        self.speech_rms_peak: float = 0.0
        self.nearfield_lock_rms: float = 0.0
        self.nearfield_lock_until: float = 0.0
        self.nonidle_since: float = 0.0
        self.barge_in_hold_logged: bool = False
        self.turn_text_sent: bool = False
        self.submitting: bool = False
        self.last_interrupt_at: float = 0.0
        self.last_submit_at: float = 0.0
        self.output_definitions: Dict[ChatDataType, HandlerDataInfo] = {}
        self.live_asr_text: str = ""


class HandlerDuplug(HandlerBase):
    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            name="SoulXDuplug",
            config_model=DuplugConfigModel,
            load_priority=40,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config=None):
        try:
            import websocket  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "SoulX-Duplug handler requires websocket-client. "
                "Install with: pip install websocket-client"
            ) from exc
        url = DuplugConfigModel().server_url
        if isinstance(handler_config, DuplugConfigModel):
            url = handler_config.server_url
        logger.info(f"SoulX-Duplug handler loaded, server={url}")

    def create_context(self, session_context: SessionContext, handler_config=None) -> HandlerContext:
        context = DuplugContext(session_context.session_info.session_id)
        context.shared_states = session_context.shared_states
        if isinstance(handler_config, DuplugConfigModel):
            context.config = handler_config
        # Bind result callback so nonidle interrupts immediately (official
        # dialogue-system does this inside TurnTaking.process()).
        context.client = _DuplugWsClient(
            context.config.server_url,
            context.config.request_timeout,
            on_result=lambda data, rms=0.0, ctx=context: self._on_duplug_result(ctx, data, rms),
        )
        return context

    def start_context(self, session_context: SessionContext, handler_context: HandlerContext):
        context = cast(DuplugContext, handler_context)
        self._clear_utterance(context)
        logger.info(f"Duplug: session start {context.session_id}")

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        text_def = DataBundleDefinition()
        text_def.add_entry(DataBundleEntry.create_text_entry("human_text"))
        return HandlerDetail(
            inputs=[
                HandlerDataInfo(
                    type=ChatDataType.MIC_AUDIO,
                    input_consume_mode=ChatDataConsumeMode.DEFAULT,
                    input_priority=50,
                ),
                HandlerDataInfo(
                    type=ChatDataType.HUMAN_DUPLEX_AUDIO,
                    input_consume_mode=ChatDataConsumeMode.DEFAULT,
                    input_priority=50,
                ),
            ],
            outputs=[
                HandlerDataInfo(type=ChatDataType.HUMAN_TEXT, definition=text_def),
            ],
            signal_filters=[
                SignalFilterRule(ChatSignalType.WAKE_WORD, None, None),
            ],
        )

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        context = cast(DuplugContext, context)
        context.output_definitions = output_definitions
        if context.shared_states is not None and not context.shared_states.listening_enabled:
            if context.audio_buffer.size or context.live_asr_text:
                self._clear_utterance(context)
            return
        if inputs.type == ChatDataType.HUMAN_DUPLEX_AUDIO:
            self._handle_duplex_audio(context, inputs)
            return
        if inputs.type != ChatDataType.MIC_AUDIO:
            return
        if context.client is None:
            return

        audio = self._as_float_audio(inputs)
        if audio is None:
            return
        # Do not boost quiet audio: 8x gain was pulling in distant talkers
        # and burying the close-talking user in Duplug ASR.
        audio = self._gate_nearfield(context, audio)
        context.audio_buffer = np.concatenate([context.audio_buffer, audio], axis=0)
        chunk = context.config.chunk_samples
        while context.audio_buffer.size >= chunk:
            piece = context.audio_buffer[:chunk]
            context.audio_buffer = context.audio_buffer[chunk:]
            context.client.submit(context.session_id, piece)
            # State/interrupt are applied in the WS worker on_result callback.

    def _handle_duplex_audio(self, context: DuplugContext, inputs: ChatData):
        """Silero speech flags only. ASR text comes from Duplug cascade, not VAD cuts."""
        meta = {}
        if inputs.data is not None:
            meta = inputs.data.metadata or {}
        if inputs.is_first_data or meta.get("reconnected_audio") or meta.get("continue_from_stream"):
            if context.shared_states is not None:
                context.shared_states.human_speech_active = True
        if inputs.is_last_data and context.shared_states is not None:
            # Duplug speak submits the turn; Silero end only clears the speech gate.
            context.shared_states.human_speech_active = False

    def _on_duplug_result(self, context: DuplugContext, data: Dict[str, Any], rms: float = 0.0):
        """Called from WS worker thread as soon as Duplug replies."""
        if context.shared_states is not None and not context.shared_states.listening_enabled:
            return
        self._apply_duplug_state(context, data, chunk_rms=float(rms or 0.0))

    def _apply_duplug_state(
        self,
        context: DuplugContext,
        data: Optional[Dict[str, Any]] = None,
        chunk_rms: float = 0.0,
    ):
        from_infer = data is not None
        if data is None:
            if context.client is None:
                return
            data = context.client.latest or {}
        state_obj = (data.get("state") or {}) if isinstance(data, dict) else {}
        if not isinstance(state_obj, dict):
            state_obj = {}
        state = str(state_obj.get("state") or "blank")
        if state == "blank":
            return
        prev = context.last_state
        context.last_state = state
        speaking = state == "nonidle"
        complete = state == "speak"
        min_rms = float(context.config.min_eou_rms or 0.04)
        silero_speech = bool(
            context.shared_states is not None
            and context.shared_states.human_speech_active
        )
        # Only official Duplug "speak" counts as EOU. Do not treat a short
        # idle after nonidle as complete — that cuts users off mid-sentence.
        if speaking:
            if not context.saw_nonidle:
                # New utterance: drop energy from the previous turn.
                context.speech_rms_peak = 0.0
            context.saw_nonidle = True
            context.idle_after_speech = 0
            complete = False
            if chunk_rms >= min_rms:
                context.speech_rms_peak = max(context.speech_rms_peak, chunk_rms)
            if context.nonidle_since <= 0:
                context.nonidle_since = time.monotonic()
            context.turn_text_sent = False
        else:
            context.nonidle_since = 0.0
            context.barge_in_hold_logged = False
            if complete:
                context.idle_after_speech = 0
                # Duplug EOU almost always arrives on a silent trailing chunk.
                # Reject only if this utterance never had real speech energy
                # (playback bleed / empty queue), not because the last frame is quiet.
                if from_infer and not context.saw_nonidle:
                    logger.info("Duplug ignore speak (no nonidle in this utterance)")
                    complete = False
                elif (
                    from_infer
                    and context.speech_rms_peak < min_rms
                    and not silero_speech
                ):
                    logger.info(
                        f"Duplug ignore speak (utterance too quiet "
                        f"peak={context.speech_rms_peak:.4f} rms={chunk_rms:.4f})"
                    )
                    complete = False
            elif state == "idle" and context.saw_nonidle and from_infer:
                context.idle_after_speech += 1
                if chunk_rms >= min_rms:
                    context.speech_rms_peak = max(context.speech_rms_peak, chunk_rms)
        asr_text = self._clean_asr_text(
            str(
                state_obj.get("text")
                or state_obj.get("asr_buffer")
                or state_obj.get("asr_segment")
                or ""
            )
        )
        if asr_text:
            context.live_asr_text = asr_text
        if context.shared_states is not None:
            was_complete = context.shared_states.duplug_turn_complete
            context.shared_states.duplug_user_speaking = speaking
            if complete:
                context.shared_states.duplug_turn_complete = True
            elif speaking:
                context.shared_states.duplug_turn_complete = False
            now_complete = context.shared_states.duplug_turn_complete
        else:
            was_complete = False
            now_complete = complete
        if state != prev or now_complete != was_complete:
            logger.info(
                f"Duplug {prev} -> {state} complete={now_complete} asr='{asr_text}'"
            )
            latency.mark(
                "duplug", "state", session_id=context.session_id,
                from_state=prev, to_state=state, complete=now_complete,
                text=asr_text,
            )
        if speaking and from_infer:
            self._try_barge_in(context, chunk_rms)
        if complete and from_infer:
            text = asr_text or context.live_asr_text
            if self._is_junk(text, context.config.min_turn_chars):
                logger.info(f"Duplug: skip junk ASR '{text}'")
                context.saw_nonidle = False
            else:
                self._emit_human_text(context, text)
        elif (
            from_infer
            and state == "idle"
            and not complete
            and context.saw_nonidle
            and not context.turn_text_sent
            and context.idle_after_speech >= 3
        ):
            # Duplug sometimes goes idle with a full transcript before (or
            # instead of) a later silent speak. Don't wait forever.
            text = context.live_asr_text
            if not self._is_junk(text, context.config.min_turn_chars):
                logger.info(f"Duplug: idle fallback submit '{text}'")
                self._emit_human_text(context, text)

    @staticmethod
    def _clean_asr_text(text: str) -> str:
        cleaned = (text or "").strip()
        while cleaned and cleaned[0] in "，。,、；;：: ":
            cleaned = cleaned[1:].strip()
        # Paraformer cascade often inserts spaces between CJK characters.
        cleaned = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", cleaned)
        return cleaned

    def _is_junk(self, text: str, min_chars: int) -> bool:
        cleaned = (
            (text or "")
            .strip()
            .replace(",", "")
            .replace(".", "")
            .replace("，", "")
            .replace("。", "")
            .replace(" ", "")
        )
        if len(cleaned) < min_chars:
            return True
        if cleaned.lower() in BACKCHANNELS or cleaned in BACKCHANNELS:
            return True
        if cleaned in {"哼哼", "嗯哼", "哈哈哈哈"}:
            return True
        return False

    def _emit_human_text(self, context: DuplugContext, text: str):
        if context.turn_text_sent:
            return
        # Official dialogue-system pipeline_worker always stops current playback
        # before starting a new turn. Without this, barge-in text becomes the
        # "next question" while the old TTS/MuseTalk queues keep playing.
        if self._avatar_speaking(context):
            self._emit_interrupt(context, "take_turn")
        output_info = context.output_definitions.get(ChatDataType.HUMAN_TEXT)
        if output_info is None or output_info.definition is None:
            logger.warning("Duplug: HUMAN_TEXT output is not configured")
            return
        streamer = None
        if context.data_submitter is not None:
            streamer = context.data_submitter.get_streamer(ChatDataType.HUMAN_TEXT)
        if streamer is not None:
            try:
                streamer.new_stream([])
            except Exception as exc:
                logger.warning(f"Duplug: independent HUMAN_TEXT stream failed: {exc}")
        output = DataBundle(output_info.definition)
        output.set_main_data(text)
        try:
            context.submit_data(
                ChatData(type=ChatDataType.HUMAN_TEXT, data=output, is_last_data=True),
                finish_stream=True,
            )
        except Exception as exc:
            logger.warning(f"Duplug: submit HUMAN_TEXT failed: {exc}")
            return
        context.turn_text_sent = True
        context.last_submit_at = time.monotonic()
        context.live_asr_text = ""
        context.saw_nonidle = False
        context.idle_after_speech = 0
        context.speech_rms_peak = 0.0
        if context.shared_states is not None:
            context.shared_states.awaiting_avatar_response = True
            context.shared_states.duplug_committed_text = ""
            context.shared_states.duplug_turn_complete = False
            context.shared_states.duplug_user_speaking = False
        logger.info(f"Duplug: submitted LLM text '{text}'")
        latency.mark("duplug", "text_submitted", session_id=context.session_id, text=text)

    def _live_asr_text(self, context: DuplugContext) -> str:
        return self._clean_asr_text(context.live_asr_text)

    def _try_barge_in(self, context: DuplugContext, chunk_rms: float):
        """Interrupt playback only on sustained, loud, Silero-confirmed speech.

        Duplug nonidle alone is too eager: room noise and TTS echo often look
        semantic. Silero never emits INTERRUPT; it only gates this path.
        """
        if not context.config.interrupt_on_nonidle:
            return
        if not self._avatar_speaking(context):
            return
        hold_reason = None
        min_rms = float(context.config.min_interrupt_rms or 0.045)
        if chunk_rms < min_rms:
            hold_reason = f"rms={chunk_rms:.4f}<{min_rms:.4f}"
        elif context.shared_states is None or not context.shared_states.human_speech_active:
            hold_reason = "silero_not_in_speech"
        else:
            grace = float(context.config.interrupt_grace_seconds or 0.0)
            held = (
                time.monotonic() - context.nonidle_since
                if context.nonidle_since > 0
                else 0.0
            )
            if grace > 0 and held < grace:
                hold_reason = f"grace={held:.2f}s<{grace:.2f}s"
            else:
                min_chars = int(context.config.min_interrupt_chars or 0)
                if min_chars > 0:
                    text = self._live_asr_text(context)
                    if self._is_junk(text, min_chars):
                        hold_reason = f"asr='{text}'"
        if hold_reason:
            if not context.barge_in_hold_logged:
                logger.info(f"Duplug: hold barge-in ({hold_reason})")
                context.barge_in_hold_logged = True
            return
        context.barge_in_hold_logged = False
        self._emit_interrupt(context, "duplug_nonidle")

    def _emit_interrupt(self, context: DuplugContext, reason: str):
        if reason == "duplug_nonidle" and not context.config.interrupt_on_nonidle:
            return
        if not self._avatar_speaking(context):
            return
        now = time.monotonic()
        if now - context.last_interrupt_at < context.config.interrupt_cooldown_seconds:
            return
        context.last_interrupt_at = now
        context.emit_signal(
            ChatSignal(
                type=ChatSignalType.INTERRUPT,
                source_type=ChatSignalSourceType.HANDLER,
                source_name=context.owner,
                signal_data={"reason": reason},
            )
        )
        logger.info(f"Duplug: INTERRUPT ({reason})")

    def on_signal(self, context: HandlerContext, signal: ChatSignal):
        context = cast(DuplugContext, context)
        if signal.type == ChatSignalType.WAKE_WORD:
            self._clear_utterance(context)

    def _clear_utterance(self, context: DuplugContext):
        context.audio_buffer = np.zeros((0,), dtype=np.float32)
        context.live_asr_text = ""
        context.last_state = "idle"
        context.saw_nonidle = False
        context.idle_after_speech = 0
        context.speech_rms_peak = 0.0
        context.nearfield_lock_rms = 0.0
        context.nearfield_lock_until = 0.0
        context.nonidle_since = 0.0
        context.barge_in_hold_logged = False
        context.turn_text_sent = False
        context.submitting = False
        if context.shared_states is not None:
            context.shared_states.duplug_turn_complete = False
            context.shared_states.duplug_user_speaking = False
            context.shared_states.duplug_committed_text = ""

    def _as_float_audio(self, inputs: ChatData) -> Optional[np.ndarray]:
        if inputs.data is None:
            return None
        audio = inputs.data.get_main_data()
        if audio is None:
            return None
        audio = np.asarray(audio).squeeze()
        if audio.size == 0:
            return None
        audio = audio.astype(np.float32, copy=False)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.5:
            audio = audio / 32767.0
        return audio

    def _gate_nearfield(self, context: DuplugContext, audio: np.ndarray) -> np.ndarray:
        """Keep close-talk audio; replace far-field / room talk with silence."""
        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        now = time.monotonic()
        lock_on = float(context.config.nearfield_lock_rms or 0.08)
        min_feed = float(context.config.min_feed_rms or 0.03)
        if rms >= lock_on:
            context.nearfield_lock_rms = max(context.nearfield_lock_rms * 0.7, rms)
            context.nearfield_lock_until = now + float(
                context.config.nearfield_lock_seconds or 2.5
            )
        if now < context.nearfield_lock_until and context.nearfield_lock_rms > 0:
            if rms < context.nearfield_lock_rms * 0.35:
                return np.zeros_like(audio)
            return audio
        in_near_utt = (
            context.saw_nonidle
            and context.speech_rms_peak >= min_feed
        )
        if rms < min_feed and not in_near_utt:
            return np.zeros_like(audio)
        return audio

    def _avatar_speaking(self, context: DuplugContext) -> bool:
        if context.session_history is None:
            return False
        try:
            return bool(context.session_history.was_avatar_speaking_at(time.monotonic()))
        except Exception:
            return False

    def destroy_context(self, context: HandlerContext):
        context = cast(DuplugContext, context)
        self._clear_utterance(context)
        if context.client is not None:
            context.client.close()


handler_class = HandlerDuplug
