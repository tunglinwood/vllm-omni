# Copyright 2025 vLLM-Omni Team
"""Stage 1: Flow-matching audio detokenizer and vocoder."""

import json
import os
from typing import Optional

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput


class KimiAudioDiT(nn.Module):
    """Flow-matching DiT (Diffusion Transformer) for audio detokenization.

    Architecture:
    - 16 transformer layers
    - hidden_size: 2304
    - num_heads: 18
    - semantic_vocab_size: 16384
    - Input: 80 mel bins
    - Condition: 1280 (from audio tokens)
    """

    def __init__(self, config: dict):
        super().__init__()
        dit_config = config.get("model", {}).get("dit", {})

        self.hidden_size = dit_config.get("hidden_size", 2304)
        self.depth = dit_config.get("depth", 16)
        self.num_heads = dit_config.get("num_heads", 18)
        self.semantic_vocab_size = dit_config.get("semantic_vocab_size", 16384)
        self.input_size = dit_config.get("input_size", 80)  # mel bins
        self.condition_input_dim = dit_config.get("condition_input_dim", 1280)

        # Token embedding (audio tokens → embeddings)
        self.token_embed = nn.Embedding(self.semantic_vocab_size, self.hidden_size)

        # Condition prenet (transform token embeddings to hidden size)
        # Input is already hidden_size from token_embed, so we just project to hidden_size
        self.condition_prenet = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )

        # DiT layers (16 transformer layers)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_size * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(self.depth)
        ])

        # Output projection to mel spectrogram
        self.output_proj = nn.Linear(self.hidden_size, self.input_size)

        # Noise projection from mel bins to hidden size
        self.noise_proj = nn.Linear(self.input_size, self.hidden_size)

    def _get_time_embedding(self, t: float, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Sinusoidal time embedding."""
        dim = self.hidden_size
        half_dim = dim // 2
        # CUDA graph compatible: use torch.full instead of torch.tensor
        emb = torch.log(torch.full((1,), 10000.0, device=device, dtype=torch.float32)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = t * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(1, device=device)], dim=-1)
        # Convert to model dtype before passing through time_embed
        emb = emb.to(dtype)
        return self.time_embed(emb.unsqueeze(0).expand(batch_size, -1)).unsqueeze(1)

    def generate(
        self,
        audio_token_ids: torch.Tensor,
        ode_steps: int = 150,
        cfg_scale: float = 4.0,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Flow-matching inference with ODE solver.

        Args:
            audio_token_ids: [batch, seq_len] or [seq_len] - audio token IDs (152064-168447)
            ode_steps: Number of ODE integration steps (150)
            cfg_scale: Classifier-free guidance scale (4.0)
            dtype: Data type for noise generation

        Returns:
            mel_spectrogram: [batch, seq_len, 80] - mel spectrogram
        """
        # Handle 1D input by adding batch dimension
        if audio_token_ids.dim() == 1:
            audio_token_ids = audio_token_ids.unsqueeze(0)

        # Convert audio tokens to embeddings (subtract offset)
        token_ids = audio_token_ids - 152064  # Offset to 0-16383
        token_ids = token_ids.clamp(0, self.semantic_vocab_size - 1)
        token_embeds = self.token_embed(token_ids)  # [B, L, 2304]

        # Apply condition prenet
        condition = self.condition_prenet(token_embeds)

        # Initialize noise (mel spectrogram shape)
        batch_size, seq_len = audio_token_ids.shape
        noise = torch.randn(
            batch_size, seq_len, self.input_size,
            device=audio_token_ids.device,
            dtype=dtype  # Use provided dtype
        )

        # ODE integration (Euler method)
        dt = 1.0 / ode_steps
        for step in range(ode_steps):
            t = step / ode_steps

            # Predict velocity (conditional)
            velocity_cond = self._predict_velocity(noise, condition, t)

            # Classifier-free guidance
            if cfg_scale > 1.0:
                # Predict velocity (unconditional)
                velocity_uncond = self._predict_velocity(
                    noise, torch.zeros_like(condition), t
                )

                # CFG: v = v_uncond + cfg_scale * (v_cond - v_uncond)
                velocity = velocity_uncond + cfg_scale * (velocity_cond - velocity_uncond)
            else:
                velocity = velocity_cond

            # Euler step
            noise = noise + velocity * dt

        return noise

    def _predict_velocity(
        self,
        noise: torch.Tensor,
        condition: torch.Tensor,
        t: float,
    ) -> torch.Tensor:
        """Predict velocity for ODE integration."""
        batch_size, seq_len, _ = noise.shape
        device = noise.device
        dtype = noise.dtype  # Use noise dtype (should match model dtype)

        # Time embedding
        time_emb = self._get_time_embedding(t, batch_size, device, dtype)  # [B, 1, hidden]
        time_emb = time_emb.expand(-1, seq_len, -1)  # [B, L, hidden]

        # Project noise from mel bins (80) to hidden size (2304)
        # noise is [B, L, 80], need to project to [B, L, 2304]
        noise_proj = self.noise_proj(noise) if hasattr(self, 'noise_proj') else noise

        # Add all components (all should be [B, L, 2304])
        x = noise_proj + condition + time_emb  # [B, L, hidden]

        # Forward through DiT layers
        for layer in self.layers:
            x = layer(x)

        # Project to mel spectrogram
        mel = self.output_proj(x)
        return mel


class KimiAudioVocoder(nn.Module):
    """HiFi-GAN vocoder: mel spectrogram → waveform.

    Architecture:
    - Sampling rate: 24000 Hz
    - Hop size: 480
    - Upsample rates: [5, 2, 2, 2, 2, 3, 2] (total: 480x)
    - 80 mel bins
    """

    def __init__(self, config: dict):
        super().__init__()
        self.sampling_rate = config.get("sampling_rate", 24000)
        self.hop_size = config.get("hop_size", 480)
        self.num_mels = config.get("num_mels", 80)
        self.upsample_rates = config.get("upsample_rates", [5, 2, 2, 2, 2, 3, 2])
        self.upsample_kernel_sizes = config.get("upsample_kernel_sizes", [9, 4, 4, 4, 4, 5, 4])
        self.upsample_initial_channel = config.get("upsample_initial_channel", 2048)

        # Build upsampling network
        self.upsampler = self._build_upsampler()

    def _build_upsampler(self) -> nn.Module:
        """Build upsampling network with transposed convolutions."""
        layers = []
        in_channels = self.num_mels

        for i, (rate, kernel_size) in enumerate(
            zip(self.upsample_rates, self.upsample_kernel_sizes)
        ):
            out_channels = self.upsample_initial_channel // (2 ** (i + 1))
            if out_channels < 1:
                out_channels = 1

            layers.append(
                nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=rate,
                    padding=(kernel_size - rate) // 2,
                )
            )
            layers.append(nn.LeakyReLU(0.1))
            in_channels = out_channels

        # Final projection to mono audio
        layers.append(nn.Conv1d(in_channels, 1, kernel_size=7, padding=3))
        layers.append(nn.Tanh())

        return nn.Sequential(*layers)

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Convert mel spectrogram to waveform.

        Args:
            mel_spectrogram: [batch, seq_len, 80] - mel spectrogram

        Returns:
            waveform: [batch, time] - audio waveform at 24kHz
        """
        # Transpose for Conv1d: [B, L, 80] → [B, 80, L]
        mel = mel_spectrogram.transpose(1, 2)

        # Upsample: [B, 80, L] → [B, 1, L*480]
        waveform = self.upsampler(mel)

        # Squeeze: [B, 1, T] → [B, T]
        waveform = waveform.squeeze(1)

        return waveform

    @classmethod
    def from_pretrained(cls, vocoder_path: str) -> "KimiAudioVocoder":
        """Load vocoder from checkpoint."""
        config_path = os.path.join(vocoder_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        vocoder = cls(config)

        # Load weights
        weights_path = os.path.join(vocoder_path, "model.pt")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            # Handle potential state dict key mismatches
            vocoder.load_state_dict(state_dict, strict=False)

        return vocoder


class KimiAudioDetokenizerForConditionalGeneration(nn.Module):
    """Stage 1: Flow-matching DiT → vocoder → 24kHz waveform."""

    # Mark as generative model so vllm's runner validation passes
    is_text_generation_model = True
    # Mark as producing multimodal outputs (audio waveform)
    have_multimodal_outputs = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.dtype = vllm_config.model_config.dtype
        model_path = vllm_config.model_config.model

        # Load audio detokenizer config
        detokenizer_config_path = os.path.join(model_path, "audio_detokenizer", "config.yaml")
        if os.path.exists(detokenizer_config_path):
            import yaml
            with open(detokenizer_config_path, "r") as f:
                detokenizer_config = yaml.safe_load(f)
        else:
            # Fallback config
            detokenizer_config = {
                "model": {
                    "dit": {
                        "hidden_size": 2304,
                        "depth": 16,
                        "num_heads": 18,
                        "semantic_vocab_size": 16384,
                        "input_size": 80,
                    }
                },
                "ode_steps": 150,
                "cfg_scale": 4.0,
            }

        # Load vocoder config
        vocoder_config_path = os.path.join(model_path, "vocoder", "config.json")
        if os.path.exists(vocoder_config_path):
            with open(vocoder_config_path, "r") as f:
                vocoder_config = json.load(f)
        else:
            # Fallback config
            vocoder_config = {
                "sampling_rate": 24000,
                "hop_size": 480,
                "num_mels": 80,
            }

        # Initialize components
        self.dit = KimiAudioDiT(detokenizer_config)
        self.vocoder = KimiAudioVocoder.from_pretrained(
            os.path.join(model_path, "vocoder")
        )

        # Store config
        self.ode_steps = detokenizer_config.get("ode_steps", 150)
        self.cfg_scale = detokenizer_config.get("cfg_scale", 4.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        multimodal_embeddings: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        sampling_metadata: Optional[torch.Tensor] = None,
        logits_index: Optional[int] = None,
        sampler=None,
        additional_information: Optional[dict] = None,
        **kwargs,
    ) -> OmniOutput:
        """
        Convert audio tokens to waveform.

        Args:
            input_ids: [batch, seq_len] - audio tokens (152064-168447)
            positions: Position indices (not used in this stage)
            intermediate_tensors: Not used
            multimodal_embeddings: Not used
            inputs_embeds: Not used (accepted for compatibility)

        Returns:
            OmniOutput with waveform
        """
        # 1. Flow-matching inference (150 ODE steps, CFG=4.0)
        mel_spectrogram = self.dit.generate(
            input_ids,
            ode_steps=self.ode_steps,
            cfg_scale=self.cfg_scale,
            dtype=self.dtype,
        )

        # 2. Vocoder: mel → waveform
        waveform = self.vocoder(mel_spectrogram)

        # 3. Return waveform in multimodal_outputs for proper audio extraction
        # Use "model_outputs" key to match MiMo Audio pattern
        # Reshape to (1, -1) for consistency
        waveform_flat = waveform.reshape(1, -1) if waveform is not None else waveform
        # Return sample rate as scalar to avoid accumulation issues
        sr_value = 24000
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": waveform_flat,
                "sr": sr_value,  # Return as scalar int, not tensor
            },
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embed input IDs (not used for detokenizer, stub for Protocol)."""
        # Detokenizer doesn't need embeddings, return dummy
        return torch.zeros(1, device=input_ids.device)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Compute logits (not used for detokenizer, stub for Protocol)."""
        # Detokenizer doesn't compute logits, return None
        return None

    def load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        """Load weights from audio_detokenizer/ and vocoder/ subfolders."""
        model_path = self.config._name_or_path

        # Load DiT weights
        dit_weights_path = os.path.join(model_path, "audio_detokenizer", "model.pt")
        if os.path.exists(dit_weights_path):
            dit_state_dict = torch.load(dit_weights_path, map_location="cpu", weights_only=True)
            self.dit.load_state_dict(dit_state_dict, strict=False)

        # Vocoder weights loaded in __init__ via from_pretrained
