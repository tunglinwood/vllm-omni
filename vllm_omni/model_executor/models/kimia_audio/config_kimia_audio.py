# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

from transformers import Qwen2Config


@dataclass
class KimiaAudioConfig(Qwen2Config):
    """Config for Kimi-Audio TTS model.

    Kimi-Audio uses a unified vocabulary (168448 = 152064 text + 16384 audio)
    with MIMO bifurcation at layer 21 of the 28-layer Qwen2 backbone.
    """

    def __init__(
        self,
        *,
        kimia_mimo_layers: int = 6,
        kimia_mimo_transformer_from_layer_index: int = 21,
        kimia_mimo_audiodelaytokens: int = 5,
        kimia_audio_output_vocab: int = 16384,
        kimia_token_offset: int = 152064,
        kimia_text_output_vocab: int = 152064,
        kimia_adaptor_input_dim: int = 5120,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kimia_mimo_layers = kimia_mimo_layers
        self.kimia_mimo_transformer_from_layer_index = kimia_mimo_transformer_from_layer_index
        self.kimia_mimo_audiodelaytokens = kimia_mimo_audiodelaytokens
        self.kimia_audio_output_vocab = kimia_audio_output_vocab
        self.kimia_token_offset = kimia_token_offset
        self.kimia_text_output_vocab = kimia_text_output_vocab
        self.kimia_adaptor_input_dim = kimia_adaptor_input_dim

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"kimia_mimo_layers={self.kimia_mimo_layers!r}, "
            f"kimia_mimo_transformer_from_layer_index={self.kimia_mimo_transformer_from_layer_index!r}, "
            f"kimia_mimo_audiodelaytokens={self.kimia_mimo_audiodelaytokens!r}, "
            f"kimia_audio_output_vocab={self.kimia_audio_output_vocab!r}, "
            f"kimia_token_offset={self.kimia_token_offset!r}, "
            f"kimia_text_output_vocab={self.kimia_text_output_vocab!r}, "
            f"kimia_adaptor_input_dim={self.kimia_adaptor_input_dim!r})"
        )
