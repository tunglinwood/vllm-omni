# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Configuration classes for Kimi-Audio TTS models."""

from transformers import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class KimiAudioTalkerConfig(PretrainedConfig):
    r"""
    Configuration class for Kimi-Audio Talker model.

    This stores the architecture parameters for the TTS generation path:
    - Shared backbone (layers 0-21)
    - MIMO layers (0-5) for audio generation
    - Audio output head

    Args:
        hidden_size (`int`, *optional*, defaults to 3584):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 18944):
            Dimension of the MLP representations in the backbone.
        num_hidden_layers (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer backbone.
        num_attention_heads (`int`, *optional*, defaults to 28):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            Number of key/value heads for GQA.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the RMS normalization layers.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            The base period of the RoPE embeddings.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            Maximum sequence length.
        vocab_size (`int`, *optional*, defaults to 168448):
            Total vocabulary size (text + audio tokens).
            Both text and audio output heads use this full vocab size.
        text_output_vocab (`int`, *optional*, defaults to 152064):
            Vocabulary size for text output (tokens 0-152063).
            Note: text lm_head outputs full vocab, text tokens are [0:152064]
        audio_output_vocab (`int`, *optional*, defaults to 16384):
            Vocabulary size for audio output (tokens 152064-168447).
            Note: mimo_output outputs full vocab, audio tokens are [152064:168448]
        audio_token_offset (`int`, *optional*, defaults to 152064):
            Offset for audio tokens in the vocabulary.
        mimo_layers (`int`, *optional*, defaults to 6):
            Number of MIMO (audio generation) layers.
        mimo_transformer_from_layer_index (`int`, *optional*, defaults to 21):
            Layer index where bifurcation happens (clone hidden states).
        kimia_adaptor_input_dim (`int`, *optional*, defaults to 5120):
            Input dimension for VQ-Adaptor (Whisper features).
        use_whisper_feature (`bool`, *optional*, defaults to `True`):
            Whether to use Whisper features for audio input.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialization.
    """

    model_type = "kimi_audio_tts"

    def __init__(
        self,
        hidden_size: int = 3584,
        intermediate_size: int = 18944,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 28,
        num_key_value_heads: int = 4,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-06,
        rope_theta: float = 1000000.0,
        max_position_embeddings: int = 8192,
        vocab_size: int = 168448,
        text_output_vocab: int = 152064,
        audio_output_vocab: int = 16384,
        audio_token_offset: int = 152064,
        mimo_layers: int = 6,
        mimo_transformer_from_layer_index: int = 21,
        kimia_adaptor_input_dim: int = 5120,
        use_whisper_feature: bool = True,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Backbone architecture
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size
        self.initializer_range = initializer_range

        # TTS-specific configuration
        self.text_output_vocab = text_output_vocab
        self.audio_output_vocab = audio_output_vocab
        self.audio_token_offset = audio_token_offset
        self.mimo_layers = mimo_layers
        self.mimo_transformer_from_layer_index = mimo_transformer_from_layer_index
        self.kimia_adaptor_input_dim = kimia_adaptor_input_dim
        self.use_whisper_feature = use_whisper_feature

        # Special token IDs (from Kimi-Audio tokenizer)
        self.media_begin_token_id = 151661
        self.media_end_token_id = 151663
        self.text_blank_token_id = 151666


class KimiAudioTTSConfig(PretrainedConfig):
    r"""
    Configuration class for Kimi-Audio TTS system.

    This is the top-level configuration that contains:
    - talker_config: Configuration for the talker model
    - Model paths and other system-level settings

    Args:
        talker_config (`dict` or `KimiAudioTalkerConfig`, *optional*):
            Configuration for the talker model.
        model_path (`str`, *optional*):
            Path to the Kimi-Audio model checkpoint.
        audio_detokenizer_path (`str`, *optional*):
            Path to the audio detokenizer.
    """

    model_type = "kimi_audio_tts_system"

    def __init__(
        self,
        talker_config: dict | KimiAudioTalkerConfig | None = None,
        model_path: str | None = None,
        audio_detokenizer_path: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(talker_config, dict):
            talker_config = KimiAudioTalkerConfig(**talker_config)
        elif talker_config is None:
            talker_config = KimiAudioTalkerConfig()

        self.talker_config = talker_config
        self.model_path = model_path
        self.audio_detokenizer_path = audio_detokenizer_path
