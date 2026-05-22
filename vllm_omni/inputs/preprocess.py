from typing import Any

from typing_extensions import assert_never
from vllm.inputs import EmbedsInput, MultiModalInput, SingletonInput
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.renderers.inputs import SingletonDictPrompt

from vllm_omni.inputs.data import (
    OmniEmbedsPrompt,
    OmniTextPrompt,
    OmniTokenInputs,
    OmniTokensPrompt,
    token_inputs_omni,
)

logger = init_logger(__name__)

# Kimi-Audio S2S scaffolding token IDs (absolute vocab IDs).
# Reference: Kimi-Audio-7B-Instruct config.
KIMIA_USER_MSG_START = 151670
KIMIA_MEDIA_BEGIN = 151661
KIMIA_MEDIA_END = 151663
KIMIA_SPEECH_CTD = 151676
KIMIA_MSG_END = 151645
KIMIA_TOKEN_OFFSET = 152064  # audio codes start at this offset


def _build_s2s_token_sequence(
    whisper_feature,
) -> list[int]:
    """Build Kimi-Audio S2S token sequence with audio scaffolding.

    The reference model expects:
    [user_msg_start, media_begin, audio_0...audio_N, media_end, speech_ctd, msg_end]

    The number of audio tokens is derived from the Whisper feature length.
    Text tokens are NOT appended here — the dual-stream model handles text
    separately via text_input_ids. For vLLM's single-stream path, the text
    prompt's semantic content is already encoded in the whisper features.

    Args:
        whisper_feature: Whisper embedding tensor [1, audio_frames, 3584] or
            raw tensor [1, audio_frames, 5120]. The second dimension gives the
            audio frame count.

    Returns:
        Full S2S token sequence list (scaffolding only, no text).
    """
    # Determine audio frame count from whisper feature shape
    if hasattr(whisper_feature, "shape"):
        shape = whisper_feature.shape
        if isinstance(shape, (list, tuple)):
            audio_frames = shape[1] if len(shape) >= 2 else shape[0]
        else:
            audio_frames = int(shape[1]) if len(shape) >= 2 else int(shape[0])
    elif hasattr(whisper_feature, "ndim") and whisper_feature.ndim >= 2:
        audio_frames = whisper_feature.shape[1]
    else:
        logger.warning(
            "Cannot determine audio frames from whisper feature, "
            "falling back to text-only tokenization"
        )
        return None

    # Build the S2S token sequence (no text appended)
    audio_token_ids = list(
        range(KIMIA_TOKEN_OFFSET, KIMIA_TOKEN_OFFSET + audio_frames)
    )
    s2s_tokens = [
        KIMIA_USER_MSG_START,
        KIMIA_MEDIA_BEGIN,
        *audio_token_ids,
        KIMIA_MEDIA_END,
        KIMIA_SPEECH_CTD,
        KIMIA_MSG_END,
    ]

    logger.info(
        "Built S2S token sequence: %d audio + 6 scaffolding = %d total",
        audio_frames, len(s2s_tokens),
    )
    return s2s_tokens


def _extract_whisper_feature(additional_information: dict | None):
    """Extract whisper feature from additional_information if present.

    Returns the whisper tensor or None.
    """
    if not additional_information:
        return None
    for key in ("whisper_input_feature", "whisper_feat", "whisper_raw"):
        feat = additional_information.get(key)
        if feat is not None:
            return feat
    return None


class OmniInputPreprocessor(InputPreprocessor):
    """Input preprocessor for omni models.

    Extends the base InputPreprocessor to handle omni-specific input
    types including prompt embeddings and additional information payloads.
    Supports processing tokens, embeddings, text, and multimodal inputs.
    """

    def _tokenize_prompt(
        self,
        prompt: str,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[int]:
        """Tokenize text prompt, with fallback for TikTokenTokenizer.

        Some custom tokenizers (e.g. Kimi-Audio's TikTokenTokenizer) don't
        accept standard kwargs like `truncation` or `add_special_tokens`.
        Check by name first, then fall back to catching TypeError for
        unknown tokenizers.
        """
        tokenizer = self.get_tokenizer()
        # Tokenizers like TikTokenTokenizer don't support standard kwargs
        if "TikToken" in type(tokenizer).__name__:
            return tokenizer.encode(prompt)

        try:
            return super()._tokenize_prompt(prompt, tokenization_kwargs)
        except TypeError as e:
            if "unexpected keyword argument" in str(e) or "got an unexpected keyword argument" in str(e):
                logger.debug(
                    "Tokenizer doesn't support tokenization kwargs, calling directly: %s",
                    e,
                )
                return tokenizer.encode(prompt)
            raise

    def _process_text(
        self,
        parsed_content: OmniTextPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: Any | None = None,
    ) -> OmniTokenInputs | MultiModalInput:
        """Process text prompts with support for mm_processor_kwargs.

        Extends base class to support mm_processor_kwargs without multi_modal_data.
        This is needed for models like GLM-Image where text-to-image generation
        requires processor kwargs (target_h, target_w) to format the prompt.

        For Kimi-Audio S2S: when whisper features are present in additional_information,
        constructs the proper S2S token sequence with audio scaffolding tokens instead
        of plain text tokenization.
        """
        prompt_text = parsed_content["prompt"]
        mm_processor_kwargs = parsed_content.get("mm_processor_kwargs") or {}
        # When the deprecated raw-prompt path is used, process_inputs does
        # not pass mm_uuids to preprocess().  Fall back to reading it from
        # the prompt dict so the Renderer's _validate_mm_uuids can see it.
        effective_mm_uuids = mm_uuids or parsed_content.get("multi_modal_uuids")

        # Check for Kimi-Audio S2S whisper features
        additional_information = parsed_content.get("additional_information")
        whisper_feature = _extract_whisper_feature(additional_information)

        inputs: OmniTokenInputs | MultiModalInput
        if multi_modal_data := parsed_content.get("multi_modal_data"):
            inputs = self._process_multimodal(
                prompt_text,
                multi_modal_data,
                mm_processor_kwargs,
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=effective_mm_uuids,
            )
            prompt_embeds = parsed_content.get("prompt_embeds")
            if prompt_embeds is not None:
                inputs["prompt_embeds"] = prompt_embeds
            if additional_information is not None:
                inputs["additional_information"] = additional_information
        elif "mm_processor_kwargs" in parsed_content:
            # Presence — not truthiness. An explicitly-set empty dict still
            # signals "route through the multimodal processor" (needed for
            # AR-based image-gen where the HF processor supplies its own
            # defaults and scaffold).
            inputs = self._process_multimodal(
                prompt_text,
                {},
                mm_processor_kwargs,
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=effective_mm_uuids,
            )
        elif whisper_feature is not None:
            # Kimi-Audio S2S mode: build full token sequence with audio scaffolding.
            # The reference model's S2S path uses ONLY the audio scaffolding tokens
            # (no text tokens). The semantic content is encoded in the whisper features
            # injected at audio positions during embedding.
            prompt_token_ids = _build_s2s_token_sequence(whisper_feature)
            if prompt_token_ids is None:
                # Fallback to normal text tokenization if S2S build failed
                prompt_token_ids = self._tokenize_prompt(
                    prompt_text,
                    tokenization_kwargs=tokenization_kwargs,
                )
            inputs = token_inputs_omni(
                prompt_token_ids,
                prompt_embeds=parsed_content.get("prompt_embeds"),
                additional_information=additional_information,
            )
        else:
            prompt_token_ids = self._tokenize_prompt(
                prompt_text,
                tokenization_kwargs=tokenization_kwargs,
            )
            inputs = token_inputs_omni(
                prompt_token_ids,
                prompt_embeds=parsed_content.get("prompt_embeds"),
                additional_information=parsed_content.get("additional_information"),
            )
        prompt_embeds = parsed_content.get("prompt_embeds")
        if prompt_embeds is not None:
            inputs["prompt_embeds"] = prompt_embeds
        if additional_information is not None:
            inputs["additional_information"] = additional_information
        if cache_salt := parsed_content.get("cache_salt"):
            inputs["cache_salt"] = cache_salt

        return inputs

    def _process_tokens(
        self,
        parsed_content: OmniTokensPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> OmniTokenInputs | MultiModalInput:
        prompt_token_ids = self._truncate_inputs(parsed_content["prompt_token_ids"], tokenization_kwargs)
        prompt_embeds = parsed_content.get("prompt_embeds")
        additional_information = parsed_content.get("additional_information")

        multi_modal_data = parsed_content.get("multi_modal_data")

        # Check for Kimi-Audio S2S whisper features when no multi_modal_data is present.
        # If whisper features exist and the token sequence doesn't already contain
        # audio scaffolding tokens, build the proper S2S sequence.
        whisper_feature = _extract_whisper_feature(additional_information)
        has_audio_scaffolding = (
            prompt_token_ids
            and KIMIA_MEDIA_BEGIN in prompt_token_ids
            and KIMIA_MEDIA_END in prompt_token_ids
        )

        inputs: OmniTokenInputs | MultiModalInput
        if multi_modal_data:
            inputs = self._process_multimodal(
                prompt_token_ids,
                multi_modal_data,
                parsed_content.get("mm_processor_kwargs"),
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=parsed_content.get("multi_modal_uuids"),
            )
        elif whisper_feature is not None and not has_audio_scaffolding:
            # S2S mode: replace token sequence with proper audio scaffolding
            s2s_token_ids = _build_s2s_token_sequence(whisper_feature)
            if s2s_token_ids is not None:
                prompt_token_ids = s2s_token_ids
            inputs = token_inputs_omni(
                prompt_token_ids=prompt_token_ids,
                prompt_embeds=prompt_embeds,
                additional_information=additional_information,
            )
        else:
            inputs = token_inputs_omni(
                prompt_token_ids=prompt_token_ids,
                prompt_embeds=prompt_embeds,
                additional_information=additional_information,
            )
        if prompt_embeds is not None:
            inputs["prompt_embeds"] = prompt_embeds
        if additional_information is not None:
            inputs["additional_information"] = additional_information
        if prompt_text := parsed_content.get("prompt"):
            inputs["prompt"] = prompt_text
        if cache_salt := parsed_content.get("cache_salt"):
            inputs["cache_salt"] = cache_salt

        return inputs

    def _process_embeds(
        self,
        parsed_content: OmniEmbedsPrompt,
    ) -> EmbedsInput:
        """Process embeddings prompt with omni-specific extensions.

        Extends base _process_embeds to handle additional_information payload
        for direct transfer between pipeline stages.
        """
        # Call parent implementation for base embeds processing
        inputs = super()._process_embeds(parsed_content)

        # Add omni-specific additional_information if present
        additional_information = parsed_content.get("additional_information")
        if additional_information is not None:
            inputs["additional_information"] = additional_information  # type: ignore[typeddict-unknown-key]

        return inputs

    def _prompt_to_llm_inputs(
        self,
        prompt: SingletonDictPrompt,
        tokenization_kwargs: dict[str, Any] | None = None,
        *,
        mm_uuids: Any | None = None,
    ) -> SingletonInput:
        """
        Extract the singleton inputs from a prompt.

        Arguments:

        * prompt: single encoder or decoder input prompt

        Returns:

        * [`SingletonInput`][vllm.inputs.SingletonInput] instance
        """
        if "prompt_embeds" in prompt:
            return self._process_embeds(prompt)  # type: ignore[arg-type]

        if "prompt_token_ids" in prompt:
            return self._process_tokens(
                prompt,  # type: ignore[arg-type]
            )

        if "prompt" in prompt:
            return self._process_text(
                prompt,  # type: ignore[arg-type]
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )

        assert_never(prompt)  # type: ignore[arg-type]
