# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Kimi-Audio Code2Wav model - converts audio tokens to waveform."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


class KimiAudioDetokenizer(nn.Module):
    """Audio detokenizer for Kimi-Audio.
    
    Loads the audio detokenizer model.pt and converts audio token sequences
    to 24kHz audio waveforms.
    
    The detokenizer expects audio tokens in the range [152064, 168447]
    and produces waveforms at 24kHz sample rate.
    """
    
    def __init__(self, model_path: str, device: torch.device | str = "cuda"):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self._model: nn.Module | None = None
        self._sample_rate: int = 24000
        
    def load(self) -> None:
        """Load the detokenizer model."""
        if self._model is not None:
            return
            
        detokenizer_path = os.path.join(self.model_path, "audio_detokenizer", "model.pt")
        
        if not os.path.exists(detokenizer_path):
            raise FileNotFoundError(
                f"Audio detokenizer not found at {detokenizer_path}. "
                "Please ensure the Kimi-Audio model checkpoint contains the audio_detokenizer directory."
            )
        
        logger.info(f"Loading Kimi-Audio detokenizer from {detokenizer_path}")
        
        # Load the model - it's a PyTorch Lightning checkpoint
        checkpoint = torch.load(detokenizer_path, map_location=self.device, weights_only=True)
        
        # Extract actual state dict from Lightning checkpoint
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if present
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint
        
        # The detokenizer is typically a Vocoder model
        # We need to reconstruct it based on the checkpoint structure
        self._model = self._build_detokenizer(state_dict)
        self._model.load_state_dict(state_dict, strict=False)
        self._model.to(self.device)
        self._model.eval()
        
        logger.info(f"Kimi-Audio detokenizer loaded successfully (sample_rate={self._sample_rate}Hz)")
    
    def _build_detokenizer(self, state_dict: dict) -> nn.Module:
        """Build detokenizer architecture from state dict keys."""
        # Analyze state dict to determine architecture
        keys = list(state_dict.keys())
        
        # Common detokenizer architectures:
        # - HiFi-GAN: contains "generator" or "hifigan"
        # - Vocos: contains "vocos"
        # - BigVGAN: contains "bigvgan"
        # - DiT Vocoder: contains "speech_model" and "adaLN_modulation"
        # - Custom: look for patterns
        
        # For Kimi-Audio, it's a DiT-based vocoder
        # Build a decoder that can load the weights
        
        # Try to infer architecture from keys
        if any("hifigan" in k.lower() or "generator" in k.lower() for k in keys):
            return self._build_hifigan(state_dict)
        elif any("vocos" in k.lower() for k in keys):
            return self._build_vocos(state_dict)
        elif any("speech_model" in k.lower() or "adaLN_modulation" in k.lower() for k in keys):
            # Kimi-Audio uses a DiT-based vocoder
            return self._build_dit_vocoder(state_dict)
        else:
            # Generic decoder
            return self._build_generic_decoder(state_dict)
    
    def _build_hifigan(self, state_dict: dict) -> nn.Module:
        """Build HiFi-GAN style decoder."""
        from vllm_omni.model_executor.models.kimi_audio_tts.audio_detokenizer_loader import (
            HifiGanGenerator,
        )
        
        # Infer config from state dict
        # Typical HiFi-GAN has upsample_rates and upsample_kernel_sizes
        upsample_rates = [8, 8, 2, 2]  # Default
        upsample_kernel_sizes = [16, 16, 4, 4]  # Default
        resblock_kernel_sizes = [3, 7, 11]
        resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        
        model = HifiGanGenerator(
            num_mels=80,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
        )
        return model
    
    def _build_vocos(self, state_dict: dict) -> nn.Module:
        """Build Vocos style decoder."""
        from vllm_omni.model_executor.models.kimi_audio_tts.audio_detokenizer_loader import (
            VocosDecoder,
        )
        
        model = VocosDecoder(
            input_dim=80,
            hidden_dim=512,
            intermediate_dim=1536,
            num_layers=8,
        )
        return model
    
    def _build_dit_vocoder(self, state_dict: dict) -> nn.Module:
        """Build DiT-based vocoder for Kimi-Audio.
        
        This matches the architecture of audio_detokenizer/model.pt:
        - 9 transformer blocks
        - 2304 hidden dimension
        - adaLN modulation
        - 16385 vocabulary size
        """
        from vllm_omni.model_executor.models.kimi_audio_tts.audio_detokenizer_loader import (
            DiTVocoder,
        )
        
        # Infer hidden_dim from state dict
        hidden_dim = 2304  # Default for Kimi-Audio
        num_blocks = 9  # Default for Kimi-Audio
        vocab_size = 16385  # Includes padding token
        
        for key, tensor in state_dict.items():
            if "semantic_token_embedding" in key and hasattr(tensor, 'shape'):
                vocab_size, hidden_dim = tensor.shape
                logger.info(f"Inferred vocab_size={vocab_size}, hidden_dim={hidden_dim} from {key}")
                break
        
        # Count number of blocks
        block_keys = [k for k in state_dict.keys() if "speech_model.blocks." in k]
        if block_keys:
            block_indices = set()
            for k in block_keys:
                parts = k.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    block_indices.add(int(parts[2]))
            if block_indices:
                num_blocks = max(block_indices) + 1
                logger.info(f"Inferred num_blocks={num_blocks} from state dict")
        
        logger.info(
            f"Creating DiTVocoder with vocab_size={vocab_size}, "
            f"hidden_dim={hidden_dim}, num_blocks={num_blocks}"
        )
        
        model = DiTVocoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            sample_rate=self._sample_rate,
        )
        return model
    
    def _build_generic_decoder(self, state_dict: dict) -> nn.Module:
        """Build generic decoder that can load arbitrary weights."""
        from vllm_omni.model_executor.models.kimi_audio_tts.audio_detokenizer_loader import (
            GenericAudioDecoder,
        )
        
        # Infer hidden dim from state dict - CAP at 1024 to avoid OOM
        hidden_dim = 512  # Default
        for key, tensor in state_dict.items():
            if hasattr(tensor, 'shape') and len(tensor.shape) == 2:
                # Cap at 1024 to avoid OOM
                hidden_dim = min(1024, max(hidden_dim, tensor.shape[0]))
        
        # Ensure hidden_dim is divisible by 8 (for 8 attention heads)
        # Round down to nearest multiple of 8
        if hidden_dim % 8 != 0:
            hidden_dim = (hidden_dim // 8) * 8
            if hidden_dim < 64:
                hidden_dim = 64  # Minimum valid dimension
            logger.warning(f"Adjusted hidden_dim to {hidden_dim} for transformer compatibility")
        
        logger.info(f"Creating GenericAudioDecoder with hidden_dim={hidden_dim}")
        model = GenericAudioDecoder(
            vocab_size=16384,
            hidden_dim=hidden_dim,
            sample_rate=self._sample_rate,
        )
        return model
    
    @property
    def model(self) -> nn.Module:
        """Get the loaded model."""
        if self._model is None:
            self.load()
        return self._model
    
    @torch.no_grad()
    def decode(
        self,
        audio_codes: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Decode audio codes to waveform.
        
        Args:
            audio_codes: Audio token IDs [batch, seq_len] or [seq_len]
            **kwargs: Additional arguments
            
        Returns:
            waveform: Audio waveform [batch, 1, samples] or [1, samples]
        """
        if self._model is None:
            self.load()
        
        # Ensure audio_codes are in correct format
        if audio_codes.ndim == 1:
            audio_codes = audio_codes.unsqueeze(0)
        
        # Convert audio token IDs to embeddings
        # Audio tokens are in range [152064, 168447]
        # Normalize to [0, 16383] for embedding lookup
        audio_codes_normalized = audio_codes - 152064
        
        # Clamp to valid range to avoid CUDA assert
        audio_codes_normalized = torch.clamp(audio_codes_normalized, 0, 16383)
        
        # Decode
        waveform = self.model(audio_codes_normalized, **kwargs)
        
        return waveform


class KimiAudioCode2Wav(nn.Module):
    """Kimi-Audio Code2Wav stage for vLLM-Omni.
    
    Converts audio token sequences from the Talker stage into waveforms.
    
    Usage:
        Set model_stage="kimi_audio_code2wav" in vllm_config
    """
    
    input_modalities = "audio"
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True
        
        # Initialize detokenizer
        self.detokenizer = KimiAudioDetokenizer(
            model_path=self.model_path,
            device=vllm_config.device_config.device,
        )
        
        # Sample rate
        self.sample_rate = 24000  # Kimi-Audio outputs 24kHz audio
    
    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        """Dummy embedding - not used for Code2Wav."""
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> None:
        """No logits for Code2Wav - direct waveform generation."""
        return None
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """Decode audio codes to waveform.
        
        Args:
            input_ids: Audio token IDs (flattened from talker output)
            positions: Not used
            intermediate_tensors: Not used
            inputs_embeds: Not used
            runtime_additional_information: Additional info including left_context_size
            **kwargs: Additional arguments
            
        Returns:
            OmniOutput with waveform tensor
        """
        if input_ids is None or input_ids.numel() == 0:
            # Return empty audio
            empty = torch.zeros((0,), dtype=torch.float32, device=self.vllm_config.device_config.device)
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [empty],
                    "sr": [torch.tensor(self.sample_rate, dtype=torch.int32)],
                },
            )
        
        # Reshape input_ids to [batch, seq_len]
        # Input format: flattened audio codes from talker
        audio_codes = input_ids.reshape(-1).to(dtype=torch.long)
        
        # Get left context size for trimming
        left_context_size = 0
        if runtime_additional_information is not None and len(runtime_additional_information) > 0:
            left_context_size = runtime_additional_information[0].get("left_context_size", 0)
        
        # Decode to waveform
        waveform = self.detokenizer.decode(audio_codes)
        
        # Trim left context if present
        if left_context_size > 0 and waveform.shape[-1] > left_context_size * 240:  # Approximate trimming
            # Each audio code represents ~10ms at 24kHz = 240 samples
            trim_samples = left_context_size * 240
            waveform = waveform[..., trim_samples:]
        
        # Ensure correct format
        if waveform.ndim == 2:
            waveform = waveform.squeeze(0)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": [waveform.squeeze(0).squeeze(0)],  # [samples]
                "sr": [torch.tensor(self.sample_rate, dtype=torch.int32)],
            },
        )
    
    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput:
        """Wrap model outputs in OmniOutput format."""
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        
        if isinstance(model_outputs, tuple) and len(model_outputs) == 2:
            waveform, sr = model_outputs
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": waveform,
                    "sr": sr,
                },
            )
        
        if isinstance(model_outputs, torch.Tensor):
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": model_outputs,
                    "sr": torch.tensor(self.sample_rate, dtype=torch.int32),
                },
            )
        
        raise ValueError(f"Unsupported model_outputs type: {type(model_outputs)}")
    
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Code2Wav loads weights lazily from audio_detokenizer/model.pt."""
        # Weights are loaded by the detokenizer on first forward pass
        return set()
