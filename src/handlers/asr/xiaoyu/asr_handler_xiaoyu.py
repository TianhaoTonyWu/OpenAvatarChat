from typing import Dict, Optional, cast

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
from chat_engine.data_models.runtime_data.data_bundle import (
    DataBundle,
    DataBundleDefinition,
    DataBundleEntry,
)
from engine_utils.general_slicer import SliceContext, slice_data
from handlers.asr.xiaoyu.xiaoyu_asr_client import DEFAULT_API_KEY, DEFAULT_ASR_URL, recognize


class XiaoyuASRConfig(HandlerBaseConfigModel, BaseModel):
    asr_url: str = Field(default=DEFAULT_ASR_URL)
    api_key: str = Field(default=DEFAULT_API_KEY)
    timeout_seconds: float = Field(default=15.0)
    sample_rate: int = Field(default=16000)


class XiaoyuASRContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config = XiaoyuASRConfig()
        self.output_audios = []
        self.audio_slice_context = SliceContext.create_numpy_slice_context(
            slice_size=16000,
            slice_axis=0,
        )


class HandlerXiaoyuASR(HandlerBase):
    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            name="XiaoyuASR",
            config_model=XiaoyuASRConfig,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        url = DEFAULT_ASR_URL
        if isinstance(handler_config, XiaoyuASRConfig):
            url = handler_config.asr_url
        logger.info(f"XiaoyuASR loaded, url={url}")

    def create_context(self, session_context, handler_config=None):
        context = XiaoyuASRContext(session_context.session_info.session_id)
        if isinstance(handler_config, XiaoyuASRConfig):
            context.config = handler_config
        return context

    def start_context(self, session_context, handler_context):
        pass

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_text_entry("human_text"))
        return HandlerDetail(
            inputs=[HandlerDataInfo(type=ChatDataType.HUMAN_AUDIO)],
            outputs=[HandlerDataInfo(type=ChatDataType.HUMAN_TEXT, definition=definition)],
        )

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        output_definition = output_definitions.get(ChatDataType.HUMAN_TEXT).definition
        context = cast(XiaoyuASRContext, context)
        if inputs.type != ChatDataType.HUMAN_AUDIO:
            return

        audio = inputs.data.get_main_data() if inputs.data is not None else None
        if audio is not None:
            audio = audio.squeeze()
            logger.info("XiaoyuASR audio in")
            for audio_segment in slice_data(context.audio_slice_context, audio):
                if audio_segment is None or audio_segment.shape[0] == 0:
                    continue
                context.output_audios.append(audio_segment)

        if not inputs.is_last_data:
            return

        remainder_audio = context.audio_slice_context.flush()
        if remainder_audio is not None:
            if remainder_audio.shape[0] < context.audio_slice_context.slice_size:
                remainder_audio = np.concatenate(
                    [
                        remainder_audio,
                        np.zeros(shape=(context.audio_slice_context.slice_size - remainder_audio.shape[0])),
                    ]
                )
            context.output_audios.append(remainder_audio)
        if not context.output_audios:
            logger.warning("XiaoyuASR empty audio buffer")
            return
        output_audio = np.concatenate(context.output_audios)
        context.output_audios.clear()
        try:
            output_text = recognize(
                output_audio,
                sample_rate=context.config.sample_rate,
                session_id=context.session_id,
                asr_url=context.config.asr_url,
                api_key=context.config.api_key,
                timeout=context.config.timeout_seconds,
            )
        except Exception as exc:
            logger.warning(f"XiaoyuASR request failed: {exc}")
            return
        if not output_text:
            logger.warning("XiaoyuASR empty text")
            return
        output = DataBundle(output_definition)
        output.set_main_data(output_text)
        context.submit_data(output, finish_stream=True)

    def destroy_context(self, context: HandlerContext):
        pass


handler_class = HandlerXiaoyuASR
