import io
import os
import re
import time
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, Optional, Set, cast
import librosa
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field
from abc import ABC
import requests
from requests.adapters import HTTPAdapter
import json
import base64
import threading
import queue
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
from engine_utils.directory_info import DirectoryInfo
from engine_utils.latency_tracer import latency
from chat_engine.data_models.chat_signal_type import ChatSignalType
from chat_engine.data_models.chat_signal import ChatSignal, SignalFilterRule
from chat_engine.data_models.chat_stream import StreamKey, ChatStreamIdentity
from chat_engine.data_models.chat_stream_config import ChatStreamConfig


_ERROR_SPEECH_RE = re.compile(
    r"(?is)("
    r"traceback \(most recent call last\)|"
    r"\b(connectionerror|apiconnectionerror|apistatuserror|timeouterror|"
    r"httperror|oserror|readtimeout|connecttimeout)\b|"
    r"连接错误|"
    r"connection refused|"
    r"connection aborted|"
    r"connection reset|"
    r"max retries exceeded|"
    r"failed to establish|"
    r"errno \d+"
    r")"
)


def _is_error_speech(text: str) -> bool:
    if not text or not str(text).strip():
        return False
    return _ERROR_SPEECH_RE.search(str(text)) is not None


class TTSConfig(HandlerBaseConfigModel, BaseModel):
    tts_url: str = Field(default="http://14.204.16.34:6021/vox_tts")
    api_key: str = Field(default="maiyuekeji")
    voice_type: str = Field(default="zh_male")
    lang: str = Field(default="zh")
    sample_rate: int = Field(default=24000)
    stream: bool = Field(default=True)
    speedup: bool = Field(default=False)
    emotion: bool = Field(default=True)
    max_text_length: int = Field(default=500)  # 最大文本长度，超过则分段
    min_flush_chars: int = Field(default=4)
    dump_audio: bool = Field(default=False)


_SENTENCE_END_RE = re.compile(r'[。！？.!?]$')
_WEAK_TAIL_RE = re.compile(r'[，,、；;]$')
_FLUSH_PUNCT_RE = re.compile(r'[。！？.!?，,、；;]$')
_SPLIT_RE = re.compile(r'(?<=[。！？.!?，,、；;])\s*')
_STRONG_SPLIT_RE = re.compile(r'(?<=[。！？.!?])\s*')


def _speedup_form_value(value) -> Optional[str]:
    """Return a form value only when speedup is actually enabled."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else None
    text = str(value).strip()
    if not text or text.lower() in {"false", "0", "off", "no"}:
        return None
    return text


@dataclass
class CustomTTSSession:
    """Per-stream session state"""
    input_stream_id: ChatStreamIdentity
    output_stream_key: Optional[StreamKey] = None
    cancelled: bool = False
    response: Optional[requests.Response] = None
    audio_buffer: bytes = b""
    is_complete: bool = False
    data_queue: queue.Queue = field(default_factory=queue.Queue)
    processing_thread: Optional[threading.Thread] = None
    finished: bool = False
    accumulated_text: str = ""  # 累积的文本
    text_received: bool = False  # 是否已收到文本
    request_sent: bool = False  # 是否已发送请求
    next_sentence_id: int = 0
    recent_sentences: deque = field(default_factory=lambda: deque(maxlen=16))
    flushed_strong: bool = False
    latency_first_audio: bool = False


class TTSContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config = None
        self.api_links: Dict[StreamKey, CustomTTSSession] = {}
        self.dump_audio = False
        self.audio_dump_file = None

    @classmethod
    def _create_session(cls, input_stream: ChatStreamIdentity) -> CustomTTSSession:
        return CustomTTSSession(input_stream_id=input_stream)

    def handle_text_stream(self, data: ChatData, handler: 'HandlerTTS'):
        input_stream = data.stream_id
        input_stream_key = input_stream.key

        session = self.api_links.get(input_stream_key)
        if session is None:
            # 新的输入流到达，取消所有旧 session
            for old_key, old_session in list(self.api_links.items()):
                logger.info(f"TTS: Cancelling previous session for stream {old_key}")
                self._cancel_session(old_session)
            self.api_links.clear()

            session = self._create_session(input_stream)
            self.api_links[input_stream_key] = session

            # 为新的输入流创建输出流
            streamer = self.data_submitter.get_streamer(ChatDataType.AVATAR_AUDIO)
            output_stream_id = streamer.new_stream(
                sources=[session.input_stream_id],
                name="custom_tts",
                config=ChatStreamConfig(cancelable=True)
            )
            session.output_stream_key = output_stream_id.key
            self._ensure_worker(session, handler)

        # 获取文本内容（LLM 可能流式返回片段）
        text = data.data.get_main_data()
        if text is not None:
            # 过滤特殊标记
            text = re.sub(r"<\|.*?\|>", "", text)
            if _is_error_speech(text):
                logger.warning(f"TTS: drop error text, will not speak: {text[:200]}")
                text = ""
            if text.strip():
                # 累积文本并按句子拆分
                session.accumulated_text += text
                session.text_received = True
                logger.info(f"TTS: Accumulated text, total length: {len(session.accumulated_text)}")

                # First sentence: flush on comma for faster first audio.
                # After a strong ending (。！？), only split on sentence end so
                # list fragments like "贸易、教育、" are not spoken out of order.
                split_re = _STRONG_SPLIT_RE if session.flushed_strong else _SPLIT_RE
                flush_re = _SENTENCE_END_RE if session.flushed_strong else _FLUSH_PUNCT_RE
                sentences = split_re.split(session.accumulated_text)
                if sentences and not flush_re.search(sentences[-1]):
                    incomplete = sentences.pop()
                else:
                    incomplete = ''

                flushed = False
                for sent in sentences:
                    s = sent.strip()
                    if not s:
                        continue
                    if _is_error_speech(s):
                        logger.warning(f"TTS: drop error sentence, will not speak: {s[:200]}")
                        continue
                    strong_end = bool(_SENTENCE_END_RE.search(s))
                    min_chars = 2 if session.next_sentence_id == 0 else handler.min_flush_chars
                    if not strong_end and len(s) < min_chars:
                        incomplete = s + incomplete
                        continue
                    if self._enqueue_sentence(session, handler, s):
                        flushed = True

                session.accumulated_text = incomplete
                if flushed:
                    self._ensure_worker(session, handler)

        text_end = data.is_last_data

        try:
            # 当收到文本结束标志时，把剩余未完成的文本加入队列并标记结束
            if text_end:
                if session.accumulated_text.strip():
                    last_s = session.accumulated_text.strip()
                    if _is_error_speech(last_s):
                        logger.warning(f"TTS: drop final error text, will not speak: {last_s[:200]}")
                        last_s = ""
                    elif self._drop_final_leftover(session, last_s):
                        logger.info(f"TTS drop leftover after complete sentences: {last_s[:40]}")
                        last_s = ""
                    if last_s:
                        self._enqueue_sentence(session, handler, last_s)
                    session.accumulated_text = ''
                session.is_complete = True
                self._ensure_worker(session, handler)

        except Exception as e:
            logger.error(f"TTS error: {e}")
            self._cancel_session(session)
            self.api_links.pop(input_stream_key, None)

    @staticmethod
    def _drop_final_leftover(session: CustomTTSSession, last_s: str) -> bool:
        """Skip a trailing comma-list fragment flushed after the answer already ended.

        Example: spoken "...国际传播等领域都有应用。...我都可以详细讲讲。" then leftover "贸易、教育、".
        """
        if not last_s or _SENTENCE_END_RE.search(last_s):
            return False
        has_strong = session.flushed_strong or any(
            _SENTENCE_END_RE.search(prev) for prev in session.recent_sentences
        )
        if not has_strong:
            return False
        weak = bool(_WEAK_TAIL_RE.search(last_s))
        compact = re.sub(r"[\s、，,；;]", "", last_s)
        return weak or len(compact) <= 8

    def _enqueue_sentence(self, session: CustomTTSSession, handler: 'HandlerTTS', text: str) -> bool:
        s = (text or "").strip()
        if not s:
            return False
        if len(s) > handler.max_text_length:
            parts = [p.strip() for p in re.split(r'(?<=[,，;；。！？.!?])\s*', s) if p.strip()]
            if len(parts) > 1:
                enqueued = False
                for part in parts:
                    if self._enqueue_sentence(session, handler, part):
                        enqueued = True
                return enqueued
            s = s[:handler.max_text_length]
        norm = re.sub(r"\s+", " ", s)
        for prev in session.recent_sentences:
            if prev == norm or norm in prev or prev in norm:
                logger.debug(f"TTS dedupe skipped preview={s[:40]} matches prev={prev[:40]}")
                return False
        session.next_sentence_id += 1
        sid = session.next_sentence_id
        session.recent_sentences.append(norm)
        if _SENTENCE_END_RE.search(s):
            session.flushed_strong = True
        logger.info(f"TTS enqueue sid={sid} preview={s[:40]}")
        latency.mark("tts", "enqueue", session_id=self.session_id, sid=sid, preview=s[:40], chars=len(s))
        session.data_queue.put((sid, s))
        return True

    def _ensure_worker(self, session: CustomTTSSession, handler: 'HandlerTTS'):
        if session.cancelled:
            return
        if session.processing_thread and session.processing_thread.is_alive():
            return
        session.processing_thread = threading.Thread(
            target=self._tts_worker,
            args=(session, handler),
            daemon=True,
        )
        session.processing_thread.start()

    def _tts_worker(self, session: CustomTTSSession, handler: 'HandlerTTS'):
        """Worker: 按顺序从 data_queue 取出句子，调用 TTS 接口并流式提交音频到输出流，支持取消。"""
        resp = None
        try:
            streamer = self.data_submitter.get_streamer(ChatDataType.AVATAR_AUDIO)
            # 确保输出流存在
            if session.output_stream_key is None:
                logger.warning("TTS worker: no output stream configured")
                return

            while not session.cancelled:
                try:
                    # 等待下一句，超时以便检查 cancel
                    item = session.data_queue.get(timeout=0.05)
                    # item is (sid, sentence)
                    if isinstance(item, tuple) and len(item) == 2:
                        sid, sentence = item
                    else:
                        sid = None
                        sentence = item
                except Exception:
                    # 队列空，检查是否完成
                    if session.is_complete:
                        break
                    continue

                if session.cancelled:
                    break

                # 发送单句 TTS 请求并流式处理
                try:
                    files = {
                        'tts_text': (None, sentence),
                        'key': (None, handler.api_key),
                        'type': (None, handler.voice_type),
                        'stream': (None, "true" if handler.stream else "false"),
                        'lang': (None, handler.lang),
                        'emotion': (None, "true" if handler.emotion else "false"),
                    }
                    speedup = _speedup_form_value(handler.speedup)
                    if speedup:
                        files['speedup'] = (None, speedup)

                    logger.info(f"TTS worker: Requesting TTS sid={sid} preview={sentence[:40]}")
                    t_req = time.perf_counter()
                    latency.mark(
                        "tts", "request_start", session_id=self.session_id,
                        sid=sid, preview=sentence[:40],
                    )
                    http = handler.http_session or requests
                    resp = http.post(
                        handler.tts_url,
                        files=files,
                        stream=handler.stream,
                        timeout=(3.0, 60.0),
                    )
                    if resp.status_code != 200:
                        logger.error(f"TTS worker: TTS request failed {resp.status_code}: {resp.text}")
                        continue

                    # 流式读取音频并提交
                    if handler.stream:
                        for chunk in resp.iter_content(chunk_size=1024):
                            if session.cancelled:
                                break
                            if not chunk:
                                continue
                            try:
                                audio_data = np.array(np.frombuffer(chunk, dtype=np.int16)).astype(np.float32) / 32767
                                output_audio = audio_data[np.newaxis, ...]
                                if output_audio.size > 0:
                                    if not session.latency_first_audio:
                                        session.latency_first_audio = True
                                        latency.mark(
                                            "tts", "first_chunk", session_id=self.session_id,
                                            sid=sid,
                                            ms=round((time.perf_counter() - t_req) * 1000.0, 1),
                                            bytes=len(chunk),
                                        )
                                    output = DataBundle(streamer.data_definition)
                                    output.set_main_data(output_audio)
                                    self.submit_data(output)
                                    if self.dump_audio and self.audio_dump_file:
                                        self.audio_dump_file.write(chunk)
                            except Exception as e:
                                logger.warning(f"TTS worker: Error processing chunk: {e}")
                                continue

                        # 当前句子播放完毕，发送一个短静音帧作为过渡
                        if not session.cancelled:
                            logger.debug(f"TTS worker: finished sid={sid}")
                            latency.mark(
                                "tts", "sentence_done", session_id=self.session_id,
                                sid=sid,
                                ms=round((time.perf_counter() - t_req) * 1000.0, 1),
                            )
                            silence = np.zeros(shape=(1, int(handler.sample_rate * 0.02)), dtype=np.float32)
                            output = DataBundle(streamer.data_definition)
                            output.set_main_data(silence)
                            self.submit_data(output)

                    else:
                        # 非流式：一次性读取
                        audio_bytes = resp.content
                        if audio_bytes:
                            audio_data = np.array(np.frombuffer(audio_bytes, dtype=np.int16)).astype(np.float32) / 32767
                            output_audio = audio_data[np.newaxis, ...]
                            output = DataBundle(streamer.data_definition)
                            output.set_main_data(output_audio)
                            self.submit_data(output)
                            if not session.latency_first_audio:
                                session.latency_first_audio = True
                                latency.mark(
                                    "tts", "first_chunk", session_id=self.session_id,
                                    sid=sid,
                                    ms=round((time.perf_counter() - t_req) * 1000.0, 1),
                                    bytes=len(audio_bytes), stream=False,
                                )
                            logger.debug(f"TTS worker: finished non-stream sid={sid}")

                except Exception as e:
                    logger.error(f"TTS worker: exception while handling sentence: {e}")
                finally:
                    # 标记本句处理完成
                    try:
                        session.data_queue.task_done()
                    except Exception:
                        pass

            # 所有句子处理完毕，发送结束帧
            if not session.cancelled:
                output = DataBundle(streamer.data_definition)
                output.set_main_data(np.zeros(shape=(1, 240), dtype=np.float32))
                self.submit_data(output, finish_stream=True)

            logger.info("TTS worker: finished")

        except Exception as e:
            logger.error(f"TTS worker fatal error: {e}")
        finally:
            session.finished = True
            try:
                if resp:
                    resp.close()
            except Exception:
                pass

    def _start_tts_request(self, session: CustomTTSSession, handler: 'HandlerTTS', text: str):
        """启动TTS请求"""
        if not text or len(text.strip()) == 0:
            logger.warning("TTS: Empty text, skipping request")
            session.finished = True
            return
            
        try:
            # 构建请求参数
            files = {
                'tts_text': (None, text),
                'key': (None, handler.api_key),
                'type': (None, handler.voice_type),
                'stream': (None, "true" if handler.stream else "false"),
                'lang': (None, handler.lang),
                'emotion': (None, "true" if handler.emotion else "false"),
            }
            
            speedup = _speedup_form_value(handler.speedup)
            if speedup:
                files['speedup'] = (None, speedup)
            
            logger.info(f"TTS: Sending request for text length: {len(text)}, preview: {text[:50]}...")
            
            # 发起流式请求
            response = requests.post(
                handler.tts_url,
                files=files,
                stream=handler.stream,
                timeout=60
            )
            
            if response.status_code != 200:
                error_msg = f"TTS request failed with status {response.status_code}: {response.text}"
                logger.error(error_msg)
                session.finished = True
                return
            
            session.response = response
            
            # 如果使用流式，启动处理线程
            if handler.stream:
                session.processing_thread = threading.Thread(
                    target=self._process_streaming_response,
                    args=(session, handler),
                    daemon=True
                )
                session.processing_thread.start()
            else:
                # 非流式处理
                self._process_nonstreaming_response(session, handler)
                
        except Exception as e:
            logger.error(f"Failed to start TTS request: {e}")
            session.finished = True

    def _process_streaming_response(self, session: CustomTTSSession, handler: 'HandlerTTS'):
        """处理流式响应 - 在独立线程中运行"""
        try:
            streamer = self.data_submitter.get_streamer(ChatDataType.AVATAR_AUDIO)
            chunk_count = 0
            total_bytes = 0
            
            # 读取流式数据
            for chunk in session.response.iter_content(chunk_size=4096):
                if session.cancelled:
                    logger.info("TTS: Streaming cancelled")
                    break
                    
                if chunk:
                    chunk_count += 1
                    total_bytes += len(chunk)
                    
                    # 直接提交音频数据
                    try:
                        # 确保数据是有效的PCM
                        audio_data = np.array(np.frombuffer(chunk, dtype=np.int16)).astype(np.float32) / 32767
                        output_audio = audio_data[np.newaxis, ...]
                        
                        if output_audio.size > 0:
                            output = DataBundle(streamer.data_definition)
                            output.set_main_data(output_audio)
                            self.submit_data(output)
                            
                            # 如果开启了dump音频
                            if self.dump_audio and self.audio_dump_file:
                                self.audio_dump_file.write(chunk)
                    except Exception as e:
                        logger.warning(f"TTS: Error processing audio chunk: {e}")
                        continue
            
            logger.info(f"TTS: Streaming completed, received {chunk_count} chunks, {total_bytes} bytes")
            
            # 发送结束帧
            if not session.cancelled and not session.finished:
                output = DataBundle(streamer.data_definition)
                output.set_main_data(np.zeros(shape=(1, 240), dtype=np.float32))
                self.submit_data(output, finish_stream=True)
                logger.info("TTS: Sent final frame")
            
        except Exception as e:
            logger.error(f"Streaming processing error: {e}")
        finally:
            session.finished = True
            if session.response:
                try:
                    session.response.close()
                except Exception:
                    pass
                session.response = None

    def _process_nonstreaming_response(self, session: CustomTTSSession, handler: 'HandlerTTS'):
        """处理非流式响应"""
        try:
            streamer = self.data_submitter.get_streamer(ChatDataType.AVATAR_AUDIO)
            
            # 获取完整响应数据
            audio_data_bytes = session.response.content
            
            if audio_data_bytes:
                # 处理音频数据
                audio_data = np.array(np.frombuffer(audio_data_bytes, dtype=np.int16)).astype(np.float32) / 32767
                output_audio = audio_data[np.newaxis, ...]
                output = DataBundle(streamer.data_definition)
                output.set_main_data(output_audio)
                self.submit_data(output)
                
                # 发送结束帧
                output = DataBundle(streamer.data_definition)
                output.set_main_data(np.zeros(shape=(1, 240), dtype=np.float32))
                self.submit_data(output, finish_stream=True)
                
                # 如果开启了dump音频
                if self.dump_audio and self.audio_dump_file:
                    self.audio_dump_file.write(audio_data_bytes)
                
                logger.info(f"TTS: Non-streaming completed, received {len(audio_data_bytes)} bytes")
            
        except Exception as e:
            logger.error(f"Non-streaming processing error: {e}")
        finally:
            session.finished = True
            if session.response:
                try:
                    session.response.close()
                except Exception:
                    pass
                session.response = None

    def _cancel_session(self, session: CustomTTSSession):
        """取消session"""
        if session is None:
            return
        # mark cancelled early so worker can observe
        session.cancelled = True
        # drain pending queue items to avoid further TTS requests
        try:
            while not session.data_queue.empty():
                session.data_queue.get_nowait()
                try:
                    session.data_queue.task_done()
                except Exception:
                    pass
        except Exception:
            pass
        if session.response:
            try:
                session.response.close()
            except Exception:
                pass
            session.response = None
        if session.processing_thread and session.processing_thread.is_alive():
            try:
                session.processing_thread.join(timeout=5)
            except Exception:
                pass
        session.finished = True

    def _cleanup_session(self, session: CustomTTSSession):
        """清理session资源"""
        if session is None:
            return
        # 等待处理完成
        if session.processing_thread and session.processing_thread.is_alive():
            session.processing_thread.join(timeout=30)
        # 关闭响应
        if session.response:
            try:
                session.response.close()
            except Exception:
                pass
            session.response = None


class HandlerTTS(HandlerBase, ABC):
    def __init__(self):
        super().__init__()

        self.tts_url = None
        self.api_key = None
        self.voice_type = None
        self.lang = None
        self.sample_rate = None
        self.stream = None
        self.speedup = False
        self.emotion = True
        self.max_text_length = 500
        self.min_flush_chars = 4
        self.dump_audio = False
        self.http_session = requests.Session()

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=TTSConfig,
        )

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry("avatar_audio", 1, self.sample_rate))
        inputs = [
            HandlerDataInfo(type=ChatDataType.AVATAR_TEXT),
        ]
        outputs = [
            HandlerDataInfo(
                type=ChatDataType.AVATAR_AUDIO,
                definition=definition,
            )
        ]
        return HandlerDetail(
            inputs=inputs,
            outputs=outputs,
            signal_filters=[
                SignalFilterRule(ChatSignalType.STREAM_CANCEL, None, None),
                SignalFilterRule(ChatSignalType.INTERRUPT, None, None)
            ]
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        config = cast(TTSConfig, handler_config)
        self.tts_url = config.tts_url
        self.api_key = config.api_key
        self.voice_type = config.voice_type
        self.lang = config.lang
        self.sample_rate = config.sample_rate
        self.stream = config.stream
        self.speedup = bool(getattr(config, 'speedup', False))
        self.emotion = bool(getattr(config, 'emotion', True))
        self.max_text_length = getattr(config, 'max_text_length', 500)
        self.min_flush_chars = getattr(config, 'min_flush_chars', 4)
        self.dump_audio = bool(getattr(config, 'dump_audio', False))
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=4, max_retries=0)
        self.http_session.mount("http://", adapter)
        self.http_session.mount("https://", adapter)

    def create_context(self, session_context, handler_config=None):
        if not isinstance(handler_config, TTSConfig):
            handler_config = TTSConfig()
        context = TTSContext(session_context.session_info.session_id)
        context.dump_audio = bool(self.dump_audio)
        if context.dump_audio:
            dump_file_path = os.path.join(DirectoryInfo.get_project_dir(), 'temp',
                                          f"dump_avatar_audio_{context.session_id}_{int(time.time())}.pcm")
            try:
                os.makedirs(os.path.dirname(dump_file_path), exist_ok=True)
                context.audio_dump_file = open(dump_file_path, "wb")
                logger.info(f"TTS: Audio dump enabled: {dump_file_path}")
            except Exception as e:
                logger.error(f"Failed to create audio dump file: {e}")
        return context

    def start_context(self, session_context, context: HandlerContext):
        context = cast(TTSContext, context)

    def filter_text(self, text):
        pattern = r"[^a-zA-Z0-9\u4e00-\u9fff,.\~!?，。！？ ]"
        filtered_text = re.sub(pattern, "", text)
        return filtered_text

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        context = cast(TTSContext, context)
        if inputs.type == ChatDataType.AVATAR_TEXT:
            context.handle_text_stream(inputs, self)

    def on_signal(self, context: HandlerContext, signal: ChatSignal):
        """处理 STREAM_CANCEL 信号"""
        context = cast(TTSContext, context)
        # Handle stream cancel or explicit interrupt signals
        if signal.type in (ChatSignalType.STREAM_CANCEL, ChatSignalType.INTERRUPT):
            # If there's a related stream, prefer canceling that specific session
            if signal.related_stream:
                stream_key = signal.related_stream.key
                if stream_key is None:
                    return
                # 检查是否为我们的输入流被取消
                session = context.api_links.pop(stream_key, None)
                if session:
                    logger.info(f"TTS: Cancelling session for input stream {stream_key} due to {signal.type}")
                    # clear pending items
                    try:
                        while not session.data_queue.empty():
                            session.data_queue.get_nowait()
                            try:
                                session.data_queue.task_done()
                            except Exception:
                                pass
                    except Exception:
                        pass
                    context._cancel_session(session)
                    return
                # 检查是否为我们的输出流被取消
                for key, session in list(context.api_links.items()):
                    if session.output_stream_key == stream_key:
                        logger.info(f"TTS: Cancelling session for output stream {stream_key} due to {signal.type}")
                        try:
                            while not session.data_queue.empty():
                                session.data_queue.get_nowait()
                                try:
                                    session.data_queue.task_done()
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        context._cancel_session(session)
                        context.api_links.pop(key, None)
                        return
            else:
                # If no related stream is provided, treat as global interrupt: cancel all sessions
                logger.info(f"TTS: Global cancel due to {signal.type}")
                for key, session in list(context.api_links.items()):
                    try:
                        while not session.data_queue.empty():
                            session.data_queue.get_nowait()
                            try:
                                session.data_queue.task_done()
                            except Exception:
                                pass
                    except Exception:
                        pass
                    context._cancel_session(session)
                    context.api_links.pop(key, None)
                return

    def destroy_context(self, context: HandlerContext):
        context = cast(TTSContext, context)
        logger.info('destroy context')
        for session in context.api_links.values():
            context._cancel_session(session)
        context.api_links.clear()
        if context.audio_dump_file is not None:
            try:
                context.audio_dump_file.close()
            except Exception:
                pass
