from dataclasses import dataclass
from typing import Optional

from chat_engine.contexts.session_clock import SessionClock
from chat_engine.contexts.session_history import SessionHistory, HistoryConfig
from chat_engine.data_models.session_info_data import SessionInfoData


@dataclass
class SharedStates:
    active: bool = False
    # Conversation listening is on by default so configs without a wake-word
    # handler keep the existing always-on VAD behavior. WakeWordHandler sets
    # this to False when a session starts.
    listening_enabled: bool = True
    # True while duplex VAD is inside an active user utterance (START/POST_END).
    human_speech_active: bool = False
    # SoulX-Duplug: user turn is semantically complete ("speak").
    duplug_turn_complete: bool = False
    # SoulX-Duplug: current chunk has semantic content ("nonidle").
    duplug_user_speaking: bool = False
    # Official user-turn transcript from Duplug ("speak".text / asr_buffer).
    duplug_committed_text: str = ""
    # True after a user turn is sent to LLM, until avatar starts speaking.
    awaiting_avatar_response: bool = False


class SessionContext(object):
    def __init__(self, session_info: SessionInfoData, history_config: Optional[HistoryConfig] = None):
        self.session_info = session_info
        self.session_clock: SessionClock = SessionClock(self.session_info.timestamp_base)
        self.shared_states = SharedStates()
        # Global session history for full-duplex conversation support
        self.session_history: SessionHistory = SessionHistory(history_config)

    def cleanup(self):
        pass

    def get_clock(self):
        return self.session_clock
    
    def get_history(self) -> SessionHistory:
        """Get the session history for event tracking."""
        return self.session_history