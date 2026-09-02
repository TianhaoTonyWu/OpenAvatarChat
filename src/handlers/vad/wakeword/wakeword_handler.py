"""
Wake-word gate for conversation VAD.

Standby: only this handler listens (mini Silero VAD + Xiaoyu keyword check).
After a wake word: duplex VAD + Duplug open for full-duplex multi-turn
conversation. Return to standby after listen_timeout of confirmed idle
(no user speech, Duplug idle, avatar not speaking, and not waiting for
the avatar to start answering).

This handler never barge-in / interrupts. Playback interrupt is Duplug's job.
"""

import os
import re
import time
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import onnxruntime
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
from chat_engine.data_models.runtime_data.data_bundle import (
    DataBundle,
    DataBundleDefinition,
    DataBundleEntry,
)
from engine_utils.general_slicer import SliceContext, slice_data
from handlers.asr.xiaoyu.xiaoyu_asr_client import (
    DEFAULT_API_KEY,
    DEFAULT_ASR_URL,
    XiaoyuLiveSession,
    recognize as xiaoyu_recognize,
)


_PUNCT_RE = re.compile(r"[\s\.,，。！？!?\-—_、；;：:\"'“”‘’\[\]()（）【】]+")
_HOMOPHONE_MAP = {
    "语": "雨宇鱼玉",
    "雨": "语宇鱼玉",
    "宇": "语雨鱼玉",
    "鱼": "语雨宇玉",
    "玉": "语雨宇鱼",
}


class WakeWordConfigModel(HandlerBaseConfigModel, BaseModel):
    keywords: List[str] = Field(default_factory=lambda: ["你好小语"])
    asr_url: str = Field(default=DEFAULT_ASR_URL)
    api_key: str = Field(default=DEFAULT_API_KEY)
    asr_timeout_seconds: float = Field(default=8.0)
    speaking_threshold: float = Field(default=0.25)
    start_delay: int = Field(default=1024)
    end_delay: int = Field(default=8000)
    # Pure wake-word: fire after this much trailing silence once ASR already saw the keyword.
    early_wake_silence: int = Field(default=1600)  # 100ms @16kHz
    min_speech_samples: int = Field(default=3200)
    min_query_chars: int = Field(default=2)
    listen_timeout_seconds: float = Field(default=15.0)
    cooldown_seconds: float = Field(default=0.4)
    ack_text: str = Field(default="我在，有什么可以帮您。")
    ack_display_prefix: str = Field(default="【已唤醒】")
    ack_timeout_seconds: float = Field(default=5.0)


class WakeWordContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: WakeWordConfigModel = WakeWordConfigModel()
        self.shared_states = None
        self.slice_context: Optional[SliceContext] = None
        self.model_state: Optional[np.ndarray] = None
        self.speech_length: int = 0
        self.silence_length: int = 0
        self.in_speech: bool = False
        self.speech_buffer: List[np.ndarray] = []
        self.clip_size: int = 512
        self.listen_since: float = 0.0
        self.cooldown_until: float = 0.0
        self.ack_pending_listen: bool = False
        self.asr_session: Optional[XiaoyuLiveSession] = None
        self.wake_inflight: bool = False


class HandlerWakeWord(HandlerBase):
    def __init__(self):
        super().__init__()
        self.vad_model = None
        self.keywords_norm: List[Tuple[str, str]] = []

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            name="WakeWord",
            config_model=WakeWordConfigModel,
            load_priority=20,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config=None):
        model_path = os.path.abspath(
            os.path.join(
                self.handler_root,
                "..",
                "silerovad",
                "silero_vad",
                "src",
                "silero_vad",
                "data",
                "silero_vad.onnx",
            )
        )
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.log_severity_level = 4
        self.vad_model = onnxruntime.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=options,
        )
        if isinstance(handler_config, WakeWordConfigModel):
            self.keywords_norm = _compile_keywords(handler_config.keywords)
            asr_url = handler_config.asr_url
        else:
            asr_url = DEFAULT_ASR_URL
        logger.info(
            f"WakeWord loaded, keywords={[kw for kw, _ in self.keywords_norm]}, "
            f"asr={asr_url}, vad={model_path}"
        )

    def create_context(self, session_context: SessionContext, handler_config=None) -> HandlerContext:
        context = WakeWordContext(session_context.session_info.session_id)
        context.shared_states = session_context.shared_states
        if isinstance(handler_config, WakeWordConfigModel):
            context.config = handler_config
        context.slice_context = SliceContext.create_numpy_slice_context(
            slice_size=context.clip_size,
            slice_axis=0,
        )
        context.model_state = np.zeros((2, 1, 128), dtype=np.float32)
        return context

    def start_context(self, session_context: SessionContext, handler_context: HandlerContext):
        session_context.shared_states.listening_enabled = False
        session_context.shared_states.human_speech_active = False
        logger.info("WakeWord: session starts in standby (conversation VAD closed)")

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        text_def = DataBundleDefinition()
        text_def.add_entry(DataBundleEntry.create_text_entry("human_text"))
        ack_def = DataBundleDefinition()
        ack_def.add_entry(DataBundleEntry.create_text_entry("avatar_text"))
        return HandlerDetail(
            inputs=[HandlerDataInfo(type=ChatDataType.MIC_AUDIO)],
            outputs=[
                HandlerDataInfo(type=ChatDataType.HUMAN_TEXT, definition=text_def),
                HandlerDataInfo(type=ChatDataType.AVATAR_TEXT, definition=ack_def),
            ],
            signal_filters=[
                SignalFilterRule(ChatSignalType.STREAM_END, None, ChatDataType.CLIENT_PLAYBACK),
                SignalFilterRule(ChatSignalType.STREAM_CANCEL, None, ChatDataType.CLIENT_PLAYBACK),
            ],
        )

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        context = cast(WakeWordContext, context)
        if inputs.type != ChatDataType.MIC_AUDIO:
            return
        if context.shared_states is None:
            return

        now = time.monotonic()
        avatar_speaking = self._avatar_speaking(context)
        if context.ack_pending_listen:
            self._maybe_finish_ack(context, now)
        if context.shared_states.listening_enabled:
            self._maybe_timeout_listen(context, now, avatar_speaking)
            return

        # Standby only: never barge-in. Conversation interrupt is Duplug's job.
        if avatar_speaking:
            self._close_live_asr(context)
            self._reset_speech(context)
            return

        if now < context.cooldown_until:
            return

        audio = inputs.data.get_main_data() if inputs.data is not None else None
        if audio is None:
            return
        audio = audio.squeeze()
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if peak > 1.5:
                audio = audio / 32767.0
        else:
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if peak > 1.5:
                audio = audio / 32767.0

        timestamp = inputs.timestamp if inputs.is_timestamp_valid() else None
        if timestamp is not None:
            context.slice_context.update_start_id(timestamp[0], force_update=False)

        for clip in slice_data(context.slice_context, audio):
            if context.wake_inflight:
                return
            speech_prob = self._vad_prob(context, clip)
            is_speech = speech_prob >= context.config.speaking_threshold
            if is_speech:
                context.speech_length += context.clip_size
                context.silence_length = 0
                context.speech_buffer.append(np.copy(clip))
                if not context.in_speech and context.speech_length >= context.config.start_delay:
                    context.in_speech = True
                    logger.info(f"WakeWord: speech start (prob={speech_prob:.2f})")
                    self._start_live_asr(context)
                elif context.in_speech:
                    self._feed_live_asr(context, clip)
            else:
                context.silence_length += context.clip_size
                if not context.in_speech:
                    context.speech_length = 0
                    context.speech_buffer.clear()
                    self._close_live_asr(context)
                elif self._try_early_wake(context, output_definitions):
                    return
                elif context.silence_length >= context.config.end_delay:
                    self._on_idle_utterance(context, output_definitions)

    def _start_live_asr(self, context: WakeWordContext):
        self._close_live_asr(context)
        try:
            context.asr_session = XiaoyuLiveSession(
                asr_url=context.config.asr_url,
                api_key=context.config.api_key,
                continuous_decoding=True,
            )
            for clip in context.speech_buffer:
                context.asr_session.feed(clip)
            logger.info(f"WakeWord: live ASR started ({context.speech_length} samples)")
        except Exception as exc:
            logger.warning(f"WakeWord: live ASR start failed: {exc}")
            context.asr_session = None

    def _feed_live_asr(self, context: WakeWordContext, clip: np.ndarray):
        if context.asr_session is None:
            return
        try:
            context.asr_session.feed(clip)
        except Exception as exc:
            logger.warning(f"WakeWord: live ASR feed failed: {exc}")

    def _close_live_asr(self, context: WakeWordContext):
        session = context.asr_session
        context.asr_session = None
        if session is None:
            return
        try:
            session.close()
        except Exception:
            pass

    def _asr_snapshot(self, context: WakeWordContext) -> str:
        session = context.asr_session
        if session is None:
            return ""
        return (session.final_text or session.partial_text or "").strip()

    def _try_early_wake(
        self,
        context: WakeWordContext,
        output_definitions: Dict[ChatDataType, HandlerDataInfo],
    ) -> bool:
        """Wake as soon as ASR already saw a pure keyword + short trailing silence."""
        if context.silence_length < context.config.early_wake_silence:
            return False
        if context.speech_length < context.config.min_speech_samples:
            return False
        text = self._asr_snapshot(context)
        if not text:
            return False
        keyword, remainder = _match_keyword(text, self.keywords_norm)
        if keyword is None:
            return False
        # If the user kept talking after the wake word, wait for full end so
        # the trailing question is not truncated.
        if len(remainder) >= context.config.min_query_chars:
            return False
        logger.info(
            f"WakeWord: early match '{keyword}' from '{text}' "
            f"(silence={context.silence_length})"
        )
        self._close_live_asr(context)
        self._reset_speech(context)
        self._wake(context, remainder, output_definitions)
        return True

    def _maybe_timeout_listen(self, context: WakeWordContext, now: float, avatar_speaking: bool = False):
        states = context.shared_states
        if avatar_speaking:
            states.awaiting_avatar_response = False
        user_busy = bool(
            states.human_speech_active
            or avatar_speaking
            or getattr(states, "duplug_user_speaking", False)
            or getattr(states, "awaiting_avatar_response", False)
        )
        if user_busy:
            context.listen_since = now
            return
        if context.listen_since <= 0:
            context.listen_since = now
            return
        if now - context.listen_since < context.config.listen_timeout_seconds:
            return
        context.cooldown_until = now + context.config.cooldown_seconds
        context.listen_since = 0.0
        self._close_conversation_listen(
            context,
            f"user idle {context.config.listen_timeout_seconds:.0f}s, need wake word",
        )

    def _on_idle_utterance(self, context: WakeWordContext,
                           output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        clips = list(context.speech_buffer)
        speech_len = context.speech_length
        live = context.asr_session
        context.asr_session = None
        self._reset_speech(context)
        if speech_len < context.config.min_speech_samples or not clips:
            logger.info(f"WakeWord: drop short speech ({speech_len} samples)")
            if live is not None:
                try:
                    live.close()
                except Exception:
                    pass
            return
        text = ""
        try:
            if live is not None:
                text = live.finish(timeout=context.config.asr_timeout_seconds)
            else:
                audio = np.concatenate(clips, axis=0)
                text = self._asr_text(context, audio)
        except Exception as exc:
            logger.warning(f"WakeWord ASR failed: {exc}")
            if live is not None:
                try:
                    live.close()
                except Exception:
                    pass
            return
        finally:
            if live is not None:
                try:
                    live.close()
                except Exception:
                    pass
        logger.info(f"WakeWord: asr='{text}' samples={speech_len}")
        if not text:
            return
        keyword, remainder = _match_keyword(text, self.keywords_norm)
        if keyword is None:
            logger.info(f"WakeWord: ignored speech (no keyword): {text}")
            return
        logger.info(f"WakeWord: matched '{keyword}' from '{text}', remainder='{remainder}'")
        self._wake(context, remainder, output_definitions)

    def _close_conversation_listen(self, context: WakeWordContext, reason: str):
        if context.shared_states is None:
            return
        if context.shared_states.listening_enabled or context.shared_states.human_speech_active:
            logger.info(f"WakeWord: conversation VAD closed ({reason})")
        context.shared_states.listening_enabled = False
        context.shared_states.human_speech_active = False
        context.shared_states.duplug_user_speaking = False
        context.shared_states.duplug_turn_complete = False
        context.shared_states.awaiting_avatar_response = False

    def _wake(self, context: WakeWordContext, remainder: str,
              output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        if context.wake_inflight:
            return
        context.wake_inflight = True
        try:
            if len(remainder) >= context.config.min_query_chars:
                self._submit_query_text(context, remainder, output_definitions)
                self._open_listening(context, "wake with question")
                context.cooldown_until = time.monotonic() + context.config.cooldown_seconds
                return

            if not self._submit_ack(context, output_definitions):
                context.emit_signal(
                    ChatSignal(
                        type=ChatSignalType.WAKE_WORD,
                        source_type=ChatSignalSourceType.HANDLER,
                        source_name=context.owner,
                        signal_data={"remainder": remainder},
                    )
                )
                self._open_listening(context, "ack skipped")
                return

            context.emit_signal(
                ChatSignal(
                    type=ChatSignalType.WAKE_WORD,
                    source_type=ChatSignalSourceType.HANDLER,
                    source_name=context.owner,
                    signal_data={"remainder": remainder},
                )
            )
            self._open_listening(context, "woke, full duplex")
            logger.info("WakeWord: ack submitted, conversation listening opened")
        finally:
            context.wake_inflight = False
            context.cooldown_until = max(
                context.cooldown_until,
                time.monotonic() + context.config.cooldown_seconds,
            )

    def _open_listening(self, context: WakeWordContext, reason: str):
        context.shared_states.listening_enabled = True
        context.shared_states.human_speech_active = False
        context.ack_pending_listen = False
        context.listen_since = time.monotonic()
        logger.info(f"WakeWord: listening opened ({reason})")

    def _maybe_finish_ack(self, context: WakeWordContext, now: float):
        if now - context.listen_since < context.config.ack_timeout_seconds:
            return
        self._open_listening(context, "ack timeout")

    def on_signal(self, context: HandlerContext, signal: ChatSignal):
        context = cast(WakeWordContext, context)
        is_playback = (
            signal.related_stream is not None
            and signal.related_stream.data_type == ChatDataType.CLIENT_PLAYBACK
        )
        if not is_playback:
            return
        if signal.type not in (ChatSignalType.STREAM_END, ChatSignalType.STREAM_CANCEL):
            return
        if not context.ack_pending_listen:
            return
        if signal.type == ChatSignalType.STREAM_CANCEL:
            context.ack_pending_listen = False
            context.listen_since = 0.0
            context.cooldown_until = time.monotonic() + context.config.cooldown_seconds
            self._close_conversation_listen(context, "ack interrupted")
            logger.info("WakeWord: ack interrupted, stay idle (questions not accepted)")
            return
        self._open_listening(context, f"ack playback {signal.type.value}")

    def _submit_query_text(self, context: WakeWordContext, text: str,
                           output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        output_info = output_definitions.get(ChatDataType.HUMAN_TEXT)
        if output_info is None or output_info.definition is None:
            logger.warning("WakeWord: HUMAN_TEXT output is not configured")
            return
        output = DataBundle(output_info.definition)
        output.set_main_data(text)
        context.submit_data(
            ChatData(type=ChatDataType.HUMAN_TEXT, data=output, is_last_data=True),
            finish_stream=True,
        )
        if context.shared_states is not None:
            context.shared_states.awaiting_avatar_response = True

    def _submit_ack(self, context: WakeWordContext,
                    output_definitions: Dict[ChatDataType, HandlerDataInfo]) -> bool:
        text = (context.config.ack_text or "").strip()
        if not text:
            return False
        output_info = output_definitions.get(ChatDataType.AVATAR_TEXT)
        if output_info is None or output_info.definition is None:
            logger.warning("WakeWord: AVATAR_TEXT output is not configured, skip ack")
            return False
        output = DataBundle(output_info.definition)
        output.set_main_data(text)
        prefix = (context.config.ack_display_prefix or "").strip()
        if prefix:
            output.add_meta("display_prefix", prefix)
            output.add_meta("wake_ack", True)
        context.submit_data(
            ChatData(type=ChatDataType.AVATAR_TEXT, data=output, is_last_data=True),
            finish_stream=True,
        )
        logger.info(f"WakeWord: submitted ack '{prefix}{text}'")
        return True

    def _asr_text(self, context: WakeWordContext, audio: np.ndarray) -> str:
        return xiaoyu_recognize(
            audio,
            sample_rate=16000,
            session_id=context.session_id,
            asr_url=context.config.asr_url,
            api_key=context.config.api_key,
            timeout=context.config.asr_timeout_seconds,
        )

    def _vad_prob(self, context: WakeWordContext, clip: np.ndarray) -> float:
        clip = clip.squeeze()
        if clip.ndim != 1:
            return 0.0
        inputs = {
            "input": np.expand_dims(clip, axis=0),
            "sr": np.array([16000], dtype=np.int64),
            "state": context.model_state,
        }
        prob, state = self.vad_model.run(None, inputs)
        context.model_state = state
        return float(prob[0][0])

    def _reset_speech(self, context: WakeWordContext):
        context.in_speech = False
        context.speech_length = 0
        context.silence_length = 0
        context.speech_buffer.clear()
        context.model_state = np.zeros((2, 1, 128), dtype=np.float32)

    def _avatar_speaking(self, context: WakeWordContext) -> bool:
        if context.session_history is None:
            return False
        try:
            return bool(context.session_history.was_avatar_speaking_at(time.monotonic()))
        except Exception:
            return False

    def destroy_context(self, context: HandlerContext):
        context = cast(WakeWordContext, context)
        self._close_live_asr(context)
        self._reset_speech(context)


def _compile_keywords(keywords: List[str]) -> List[Tuple[str, str]]:
    compiled: List[Tuple[str, str]] = []
    seen = set()
    for kw in keywords:
        for variant in _expand_keyword_variants(kw):
            nkw = _normalize_text(variant)
            if nkw and nkw not in seen:
                seen.add(nkw)
                compiled.append((kw, nkw))
    return compiled


def _normalize_text(text: str) -> str:
    return _PUNCT_RE.sub("", (text or "")).lower()


def _expand_keyword_variants(keyword: str) -> List[str]:
    variants = {keyword}
    chars = list(keyword)
    for i, ch in enumerate(chars):
        alts = _HOMOPHONE_MAP.get(ch)
        if not alts:
            continue
        for alt in alts:
            copied = chars[:]
            copied[i] = alt
            variants.add("".join(copied))
    return list(variants)


def _match_keyword(text: str, keywords_norm: List[Tuple[str, str]]) -> Tuple[Optional[str], str]:
    norm = _normalize_text(text)
    for original, nkw in sorted(keywords_norm, key=lambda item: len(item[1]), reverse=True):
        idx = norm.find(nkw)
        if idx >= 0:
            remainder = norm[:idx] + norm[idx + len(nkw):]
            return original, remainder
    return None, norm


handler_class = HandlerWakeWord
