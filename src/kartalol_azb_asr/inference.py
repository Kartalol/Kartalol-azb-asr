"""Shared Hugging Face ASR inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from .audio import load_audio


def resolve_device(requested: str | None = None) -> str:
    """Resolve ``auto``, CPU, CUDA, or MPS with an actionable error."""

    import torch

    if requested in {None, "auto"}:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")
    return requested


def detect_model_family(config: Any) -> str:
    """Classify a Transformers config as ``whisper`` or ``ctc``."""

    model_type = str(getattr(config, "model_type", "")).lower()
    architectures = " ".join(getattr(config, "architectures", []) or []).lower()
    if model_type == "whisper" or "whisper" in architectures:
        return "whisper"
    if "ctc" in architectures or model_type in {
        "wav2vec2",
        "wav2vec2-bert",
        "hubert",
        "unispeech",
        "unispeech-sat",
    }:
        return "ctc"
    raise ValueError(
        f"Unsupported ASR architecture (model_type={model_type!r}, "
        f"architectures={architectures!r})"
    )


@dataclass
class ASRTranscriber:
    """Loaded Whisper or CTC model and its processor."""

    processor: Any
    model: Any
    family: str
    device: str
    sampling_rate: int

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device: str | None = None,
        target_lang: str = "azb",
    ) -> "ASRTranscriber":
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForCTC,
            AutoProcessor,
            WhisperForConditionalGeneration,
        )

        resolved_device = resolve_device(device)
        config = AutoConfig.from_pretrained(model_id)
        family = detect_model_family(config)
        processor = AutoProcessor.from_pretrained(model_id)
        if family == "whisper":
            model = WhisperForConditionalGeneration.from_pretrained(model_id)
        else:
            model = AutoModelForCTC.from_pretrained(model_id)
            # Base MMS needs its language adapter activated. Fine-tuned checkpoint
            # processors normally already contain the target tokenizer.
            if hasattr(processor, "tokenizer") and hasattr(
                processor.tokenizer, "set_target_lang"
            ):
                processor.tokenizer.set_target_lang(target_lang)
            if model_id == "facebook/mms-1b-all" and hasattr(model, "load_adapter"):
                model.load_adapter(target_lang)
        model = model.to(resolved_device).eval()
        sampling_rate = int(
            getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000)
        )
        if resolved_device.startswith("cuda"):
            model = model.to(dtype=torch.float16)
        return cls(processor, model, family, resolved_device, sampling_rate)

    def transcribe_array(
        self, samples: "np.ndarray", *, generation_kwargs: dict[str, Any] | None = None
    ) -> str:
        import torch

        inputs = self.processor(
            samples, sampling_rate=self.sampling_rate, return_tensors="pt"
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            if self.family == "whisper":
                generated = self.model.generate(
                    inputs["input_features"], **(generation_kwargs or {})
                )
                return self.processor.batch_decode(
                    generated, skip_special_tokens=True
                )[0].strip()
            logits = self.model(**inputs).logits
            predicted = torch.argmax(logits, dim=-1)
            return self.processor.batch_decode(predicted)[0].strip()

    def transcribe_file(
        self, path: str | Path, *, generation_kwargs: dict[str, Any] | None = None
    ) -> str:
        samples, _ = load_audio(path, self.sampling_rate)
        return self.transcribe_array(samples, generation_kwargs=generation_kwargs)
