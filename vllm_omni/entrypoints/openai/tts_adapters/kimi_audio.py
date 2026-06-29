# SPDX-License-Identifier: Apache-2.0
"""Kimi Audio serving adapter for /v1/audio/speech endpoint.

Kimi Audio is a 2-stage pipeline:
- Stage 0 (LLM): Shared backbone with bifurcation → text + audio logits
- Stage 1 (Detokenizer): Flow-matching DiT → vocoder → 24kHz waveform

The adapter handles request validation, prompt building, and audio output extraction.
"""

from typing import TYPE_CHECKING, Any

from vllm.inputs import tokens_input
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

logger = init_logger(__name__)


@register_tts_adapter
class KimiAudioTTSAdapter(ARTTSAdapter):
    """Adapter for Kimi Audio (2-stage AR + diffusion pipeline)."""

    stage_keys = frozenset({"kimi_audio"})
    name = "kimi_audio"

    def normalize(self, request: "OpenAICreateSpeechRequest") -> None:
        """Normalize request parameters."""
        # Lowercase voice if provided
        if request.voice:
            request.voice = request.voice.lower()

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        """Validate the request."""
        # Basic validation
        if not request.input:
            return "Input text is required"

        # Kimi Audio doesn't support reference audio for TTS yet
        if request.ref_audio:
            return "Kimi Audio does not support reference audio for TTS"

        return None

    async def build(
        self,
        request: "OpenAICreateSpeechRequest",
        sampling_params_list: list,
        has_inline_ref_audio: bool
    ) -> PreparedRequest:
        """Build the prompt and TTS parameters for Kimi Audio."""
        # Build TTS parameters
        tts_params = self._build_kimi_audio_params(request)

        # Build prompt (Kimi Audio uses special tokens for TTS)
        # The tokenizer will handle converting this to the proper format
        prompt_text = request.input

        # Use the tokenizer to build the prompt
        tokenizer = self.ctx.server.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            # For TTS, we need to format the prompt appropriately
            # Kimi Audio expects a chat format with audio output request
            messages = [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": "<|AUDIO|>"}
            ]

            try:
                prompt_token_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    tokenize=True,
                )
            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}, using raw text")
                prompt_token_ids = tokenizer.encode(prompt_text)
        else:
            # Fallback: just encode the text
            prompt_token_ids = tokenizer.encode(prompt_text)

        prompt = tokens_input(prompt_token_ids=prompt_token_ids)
        prompt["additional_information"] = tts_params

        return PreparedRequest(
            prompt=prompt,
            tts_params=tts_params,
            model_type=self.name,
        )

    def _build_kimi_audio_params(self, request: "OpenAICreateSpeechRequest") -> dict[str, Any]:
        """Build Kimi Audio specific parameters."""
        tts_params = {
            "task_type": ["tts"],
            "text": [request.input],
        }

        # Add voice if specified
        if request.voice:
            tts_params["voice"] = [request.voice]

        # Add response format for audio output
        tts_params["response_format"] = [request.response_format or "wav"]

        # Add sample rate
        tts_params["sample_rate"] = [24000]  # Kimi Audio outputs at 24kHz

        # Add speed if specified
        if request.speed:
            tts_params["speed"] = [request.speed]

        return tts_params
