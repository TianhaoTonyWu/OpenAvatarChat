

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Dict, Optional, Set, cast
from loguru import logger
from pydantic import BaseModel, Field
from abc import ABC
from openai import OpenAI
from engine_utils.latency_tracer import latency
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_signal import ChatSignal, SignalFilterRule
from chat_engine.data_models.chat_signal_type import ChatSignalType
from chat_engine.data_models.chat_stream import StreamKey
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
from handlers.llm.openai_compatible.chat_history_manager import filter_text
from chat_engine.data_models.chat_stream_config import ChatStreamConfig


class LLMConfig(HandlerBaseConfigModel, BaseModel):
    model_name: str = Field(default="qwen-plus")
    system_prompt: str = Field(default="请你扮演一个 AI 助手，用简短的对话来回答用户的问题，并在对话内容中加入合适的标点符号，不需要加入标点符号相关的内容")
    api_key: str = Field(default=os.getenv("DASHSCOPE_API_KEY"))
    api_url: str = Field(default=None)
    enable_video_input: bool = Field(default=False)
    history_length: int = Field(default=20)  # unused; Hermes backend session holds history
    max_tokens: int = Field(default=96)
    session_id: str = Field(default="open_avatar_chat_session")
    warmup: bool = Field(default=False)


class LLMContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config = None
        self.local_session_id = 0
        self.model_name = None
        self.system_prompt = None
        self.api_key = None
        self.api_url = None
        self.client = None
        self.input_texts = ""
        self.output_texts = ""
        self.current_image = None
        self.enable_video_input = False
        self.max_tokens = 96
        self.active_stream_keys: Set[StreamKey] = set()
        self.llm_session_id = "open_avatar_chat_session"


class HandlerLLM(HandlerBase, ABC):
    def __init__(self):
        super().__init__()
        self._session_id = "open_avatar_chat_session"
        self._warmup_enabled = False
        self._api_url = None
        self._api_key = None
        self._model_name = None

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=LLMConfig,
        )

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_text_entry("avatar_text"))
        notify_definition = DataBundleDefinition()
        notify_definition.add_entry(DataBundleEntry.create_text_entry("system_notify"))
        inputs = {
            ChatDataType.HUMAN_TEXT: HandlerDataInfo(
                type=ChatDataType.HUMAN_TEXT,
            ),
            ChatDataType.CAMERA_VIDEO: HandlerDataInfo(
                type=ChatDataType.CAMERA_VIDEO,
            ),
        }
        outputs = {
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(
                type=ChatDataType.AVATAR_TEXT,
                definition=definition,
            ),
            ChatDataType.SYSTEM_NOTIFY: HandlerDataInfo(
                type=ChatDataType.SYSTEM_NOTIFY,
                definition=notify_definition,
            ),
        }
        return HandlerDetail(
            inputs=inputs, 
            outputs=outputs,
            signal_filters=[
                SignalFilterRule(ChatSignalType.STREAM_CANCEL, None, None),
                SignalFilterRule(ChatSignalType.INTERRUPT, None, None),
            ]
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        if isinstance(handler_config, LLMConfig):
            if handler_config.api_key is None or len(handler_config.api_key) == 0:
                error_message = 'api_key is required in config/xxx.yaml, when use handler_llm'
                logger.error(error_message)
                raise ValueError(error_message)
            self._session_id = handler_config.session_id
            self._warmup_enabled = handler_config.warmup
            self._api_url = handler_config.api_url
            self._api_key = handler_config.api_key
            self._model_name = handler_config.model_name
            if self._warmup_enabled:
                threading.Thread(
                    target=self._warmup_llm,
                    args=("load",),
                    name="llm-warmup",
                    daemon=True,
                ).start()

    def create_context(self, session_context, handler_config=None):
        if not isinstance(handler_config, LLMConfig):
            handler_config = LLMConfig()
        context = LLMContext(session_context.session_info.session_id)
        context.model_name = handler_config.model_name
        context.system_prompt = {'role': 'system', 'content': handler_config.system_prompt}
        context.api_key = handler_config.api_key
        context.api_url = handler_config.api_url
        context.enable_video_input = handler_config.enable_video_input
        context.max_tokens = handler_config.max_tokens
        rtc_id = session_context.session_info.session_id
        context.llm_session_id = f"{handler_config.session_id}_{rtc_id}"
        context.client =    OpenAI(  
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key=context.api_key,
            base_url=context.api_url,
            timeout=30.0,  # 30秒超时，避免 API 无响应时阻塞整个系统
        )
        return context
    
    def start_context(self, session_context, handler_context):
        if self._warmup_enabled:
            ctx = cast(LLMContext, handler_context)
            self._warmup_llm("start", session_id=ctx.llm_session_id)

    def warmup_context(self, session_context, handler_context):
        if self._warmup_enabled:
            ctx = cast(LLMContext, handler_context)
            self._warmup_llm("session", session_id=ctx.llm_session_id)

    def _warmup_llm(self, reason: str, session_id: Optional[str] = None):
        if not self._api_url:
            return
        session_id = session_id or self._session_id
        warmup_url = self._api_url.rstrip("/") + "/warmup"
        body = json.dumps({"session_id": session_id}).encode("utf-8")
        logger.info(f"LLM warmup ({reason}): ensuring Hermes session {session_id}")
        last_err: Optional[Exception] = None
        for attempt in range(1, 6):
            req = urllib.request.Request(
                warmup_url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key or 'dummy'}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    payload = resp.read().decode("utf-8", errors="replace")
                logger.info(f"LLM warmup ({reason}) ok: {payload}")
                return
            except urllib.error.HTTPError as e:
                err_body = b""
                try:
                    err_body = e.read()
                except Exception:
                    pass
                if e.code not in (404, 405):
                    logger.warning(
                        f"LLM warmup ({reason}) failed: HTTP {e.code} {err_body[:200]!r}"
                    )
                    return
                logger.info(f"LLM warmup ({reason}): /warmup not found, falling back to a dummy completion")
                last_err = e
                break
            except Exception as e:
                last_err = e
                if attempt < 5:
                    wait_s = 2 * attempt
                    logger.warning(
                        f"LLM warmup ({reason}) attempt {attempt}/5 failed: {e}; retry in {wait_s}s"
                    )
                    time.sleep(wait_s)
                    continue
                logger.warning(f"LLM warmup ({reason}) failed: {e}")
                return
        if last_err is None:
            return
        try:
            client = OpenAI(
                api_key=self._api_key,
                base_url=self._api_url,
                timeout=60.0,
            )
            client.chat.completions.create(
                model=self._model_name or "hermes",
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
                extra_body={"session_id": session_id},
            )
            client.close()
            logger.info(f"LLM warmup ({reason}) dummy completion ok")
        except Exception as e:
            logger.warning(f"LLM warmup ({reason}) dummy completion failed: {e}")

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        output_definition = output_definitions.get(ChatDataType.AVATAR_TEXT).definition
        context = cast(LLMContext, context)

        streamer = context.data_submitter.get_streamer(ChatDataType.AVATAR_TEXT)
        if inputs.type == ChatDataType.CAMERA_VIDEO and context.enable_video_input:
            context.current_image = inputs.data.get_main_data()
            return
        elif inputs.type == ChatDataType.HUMAN_TEXT:
            text = inputs.data.get_main_data()
        else:
            return

        stream_key = streamer.current_stream.identity.stream_key_str if streamer.current_stream is not None else None
        if stream_key is None:
            stream = streamer.new_stream(sources=[inputs.stream_id], name="openai_compatible", config=ChatStreamConfig(cancelable=True))
            stream_key = stream.stream_key_str

        if text is not None:
            context.input_texts += text

        text_end = inputs.is_last_data
        if not text_end:
            return

        chat_text = context.input_texts
        chat_text = re.sub(r"<\|.*?\|>", "", chat_text)
        if len(chat_text) < 1:
            logger.warning("LLM got empty query, return emtpy response.")
            end_output = DataBundle(output_definition)
            end_output.set_main_data('')
            streamer.stream_data(end_output, name="openai_compatible", config=ChatStreamConfig(cancelable=True), finish_stream=True)
            return
        logger.info(f'llm input {context.model_name} {chat_text} ')
        llm_t0 = time.perf_counter()
        first_token = True
        latency.mark("llm", "request_start", session_id=context.session_id,
                     model=context.model_name, text=chat_text, stream=True)
        user_text = filter_text(chat_text)
        user_message = {"role": "user", "content": user_text}
        if context.enable_video_input and context.current_image is not None:
            from engine_utils.media_utils import ImageUtils
            user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": ImageUtils.format_image(context.current_image)},
                    },
                ],
            }
        messages = [context.system_prompt, user_message]
        logger.debug(f'llm input {context.model_name} {user_text} max_tokens={context.max_tokens}')
        if stream_key:
            context.active_stream_keys.add(stream_key)
        cancelled = False
        llm_failed = False
        try:
            completion = context.client.chat.completions.create(
                model=context.model_name,
                messages=messages,
                max_tokens=context.max_tokens,
                stream=True,
                stream_options={"include_usage": True},
                extra_body={"session_id": context.llm_session_id},
            )
            
            context.current_image = None
            context.input_texts = ''
            context.output_texts = ''
            cancelled = False
            for chunk in completion:
                if stream_key and stream_key not in context.active_stream_keys:
                        cancelled = True
                        try:
                            completion.close()
                        except Exception:
                            pass
                        break
                if not chunk or not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason == "error":
                    llm_failed = True
                    self._notify_frontend(
                        context, output_definitions, RuntimeError("llm stream error")
                    )
                    break
                if choice.delta and choice.delta.content:
                    output_text = choice.delta.content
                    if first_token:
                        first_token = False
                        latency.mark(
                            "llm", "first_token", session_id=context.session_id,
                            ms=round((time.perf_counter() - llm_t0) * 1000.0, 1),
                            preview=output_text[:40],
                        )
                    context.output_texts += output_text
                    logger.info(output_text)
                    output = DataBundle(output_definition)
                    output.set_main_data(output_text)
                    streamer.stream_data(output)
            if not cancelled and not llm_failed:
                if not (context.output_texts or "").strip():
                    # Backend returned no speakable content (e.g. billing).
                    # Show on frontend; do not enqueue TTS speech.
                    llm_failed = True
                    self._notify_frontend(
                        context, output_definitions, RuntimeError("empty llm response")
                    )
                else:
                    latency.mark(
                        "llm", "complete", session_id=context.session_id,
                        ms=round((time.perf_counter() - llm_t0) * 1000.0, 1),
                        chars=len(context.output_texts),
                    )
        except Exception as e:
            llm_failed = True
            logger.error(f"LLM failed, skip TTS: {e}")
            self._notify_frontend(context, output_definitions, e)
        context.input_texts = ''
        context.output_texts = ''
        if cancelled:
            return
        if stream_key:
            context.active_stream_keys.discard(stream_key)
        # On failure, finish the text stream empty so downstream closes without speaking.
        end_output = DataBundle(output_definition)
        end_output.set_main_data('')
        streamer.stream_data(end_output, finish_stream=True)

    def _frontend_error_message(self, exc: Exception) -> str:
        name = type(exc).__name__.lower()
        detail = str(exc).lower()
        if (
            "402" in detail
            or "insufficient balance" in detail
            or "credits exhausted" in detail
            or "billing" in detail
            or "empty llm response" in detail
        ):
            return "大模型账户余额不足，暂时无法回答，请稍后再试"
        if "connection" in name or "timeout" in name or "connection" in detail or "timed out" in detail:
            return "大模型暂时连不上，请稍后再试"
        return "回答失败，请稍后再试"

    def _notify_frontend(self, context: LLMContext, output_definitions, exc: Exception):
        notify_info = output_definitions.get(ChatDataType.SYSTEM_NOTIFY)
        if notify_info is None or context.data_submitter is None:
            return
        try:
            notify_streamer = context.data_submitter.get_streamer(ChatDataType.SYSTEM_NOTIFY)
            output = DataBundle(notify_info.definition)
            output.set_main_data(self._frontend_error_message(exc))
            notify_streamer.stream_data(output, finish_stream=True)
        except Exception as notify_exc:
            logger.warning(f"Failed to send frontend error notify: {notify_exc}")

    def on_signal(self, context: HandlerContext, signal: ChatSignal):
        context = cast(LLMContext, context)
        if signal.type == ChatSignalType.INTERRUPT:
            if context.active_stream_keys:
                logger.info(
                    f"LLM: INTERRUPT, drop {len(context.active_stream_keys)} active stream(s)"
                )
                context.active_stream_keys.clear()
            return
        if signal.type == ChatSignalType.STREAM_CANCEL and signal.related_stream:
            stream_key = signal.related_stream.stream_key_str
            if stream_key is not None and stream_key in context.active_stream_keys:
                context.active_stream_keys.discard(stream_key)
                logger.info(f"LLM: Removed stream {stream_key} from active set")

    def destroy_context(self, context: HandlerContext):
        context = cast(LLMContext, context)
        if context.client is not None:
            try:
                context.client.close()
            except Exception:
                pass
            context.client = None

