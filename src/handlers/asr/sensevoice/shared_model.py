import os
import threading
from typing import Any, Iterable, List, Optional

from loguru import logger

from engine_utils.directory_info import DirectoryInfo

DEFAULT_ASR_MODEL = "paraformer-zh"
_SEACO_HUB = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
_MODEL_ALIASES = {
    "paraformer-zh": _SEACO_HUB,
    "seaco": _SEACO_HUB,
}

_lock = threading.Lock()
_model = None
_model_key: Optional[str] = None
_hotwords: List[str] = []
_hotword_arg: Optional[str] = None


def normalize_hotwords(hotwords: Optional[Iterable[str]]) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in hotwords or []:
        word = (item or "").strip()
        if not word or word in seen:
            continue
        seen.add(word)
        result.append(word)
    return result


def _write_hotword_file(words: List[str]) -> Optional[str]:
    if not words:
        return None
    path = os.path.join(DirectoryInfo.get_models_dir(), "asr_hotwords.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for word in words:
            handle.write(f"{word}\n")
    return path


def configure_hotwords(hotwords: Optional[Iterable[str]] = None, merge: bool = True):
    """Register FunASR native hotwords (one phrase per line, compiled at load)."""
    global _hotwords, _hotword_arg
    incoming = normalize_hotwords(hotwords)
    with _lock:
        if merge:
            seen = set(_hotwords)
            for word in incoming:
                if word not in seen:
                    _hotwords.append(word)
                    seen.add(word)
        else:
            _hotwords = incoming
        _hotword_arg = _write_hotword_file(_hotwords)
        logger.info(f"ASR FunASR hotwords={_hotwords}, file={_hotword_arg}")


def resolve_asr_model_name(model_name: str) -> str:
    hub_id = _MODEL_ALIASES.get(model_name, model_name)
    models_dir = DirectoryInfo.get_models_dir()
    slug = hub_id.replace("/", "--")
    candidates = [
        os.path.join(models_dir, hub_id),
        os.path.join(models_dir, slug),
        os.path.join(models_dir, "models", hub_id),
        os.path.join(models_dir, "models", slug),
        os.path.join(models_dir, "models", slug, "snapshots", "master"),
        os.path.join(models_dir, os.path.basename(hub_id)),
    ]
    for path in candidates:
        if os.path.isdir(path) and (
            os.path.isfile(os.path.join(path, "model.pt"))
            or os.path.isfile(os.path.join(path, "model.onnx"))
            or os.path.isfile(os.path.join(path, "config.yaml"))
        ):
            return path
    return hub_id


def get_asr_model(model_name: str):
    """Load one FunASR ASR model per process (wake-word and ASR share weights)."""
    global _model, _model_key
    resolved = resolve_asr_model_name(model_name)
    with _lock:
        if _model is None or _model_key != resolved:
            from funasr import AutoModel
            logger.info(f"Loading shared FunASR model {resolved}")
            _model = AutoModel(model=resolved, disable_update=True)
            _model_key = resolved
        return _model


def asr_generate(audio, **kwargs) -> Any:
    if _model is None:
        raise RuntimeError("ASR model is not loaded")
    with _lock:
        if _hotword_arg and "hotword" not in kwargs:
            kwargs["hotword"] = _hotword_arg
        return _model.generate(input=audio, **kwargs)


# Back-compat names used by existing handlers.
resolve_sensevoice_model_name = resolve_asr_model_name
get_sensevoice_model = get_asr_model
sensevoice_generate = asr_generate
