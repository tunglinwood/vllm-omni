# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimia Audio code2wav stage: DiT detokenizer for TTS waveform generation.

This module implements the code2wav stage for Kimi-Audio TTS. It takes audio
code token IDs from the AR stage and decodes them into waveforms using the
DiT-based audio detokenizer followed by a HiFi-GAN vocoder.

Detokenizer architecture (from audio_detokenizer/config.yaml):
- DiT: 16 layers, hidden=2304, 18 heads, ff_mult=4
- Condition: discrete_codes (semantic_vocab_size=16384)
- Input: 80 mel bins + condition prenet
- Generation: ODE-based (150 steps) with CFG (scale=4.0)
- Output: 24kHz audio waveform

The detokenizer weights are stored as a single model.pt file (~18GB) in
the audio_detokenizer/ subdirectory of the model path.
The vocoder weights are stored in vocoder/model.pt.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm, weight_norm
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# DiT Components — named to match checkpoint keys exactly
# ---------------------------------------------------------------------------

# The checkpoint uses adaLN_modulation.1.weight [13824, 2304] = [6*2304, 2304]
# This means adaLN_modulation is a Sequential[SiLU, Linear] where Linear: dim→6*dim
# But AdaLayerNormZero from diffusers produces keys like:
#   adaLN_modulation.norm.*, adaLN_modulation.silu.*, adaLN_modulation.linear.*
# NOT adaLN_modulation.1.*
# So we CANNOT use diffusers AdaLayerNormZero — we need our own.


class KimiAudioDiTBlock(nn.Module):
    """DiT block matching Kimi-Audio checkpoint exactly."""

    def __init__(self, dim: int, heads: int, dim_head: int, ff_mult: int = 4):
        super().__init__()
        self.heads = heads
        self.inner_dim = dim_head * heads

        # Checkpoint: adaLN_modulation.1.weight → Sequential[SiLU, Linear]
        # Keys: adaLN_modulation.0 (SiLU, no params), adaLN_modulation.1 (Linear)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        self.adaLN_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        self.attn = nn.ModuleDict(
            {
                "qkv": nn.Linear(dim, 3 * self.inner_dim, bias=True),
                "proj": nn.Linear(self.inner_dim, dim, bias=True),
            }
        )

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        # Checkpoint: mlp.fc1.weight (direct Linear, not Sequential)
        self.mlp = nn.ModuleDict(
            {
                "fc1": nn.Linear(dim, dim * ff_mult),
                "fc2": nn.Linear(dim * ff_mult, dim),
            }
        )
        self.mlp_act = nn.GELU(approximate="tanh")

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        incremental_state: dict | None = None,
        rotary_pos_emb: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Combined embedding c = t_embedder(t) + condition [B, T, D]
        # Pass through adaLN modulation to get shift/scale/gate params
        emb = self.adaLN_modulation(c)  # [B, T, 6*dim]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)

        norm_x = self.adaLN_norm(x) * (1 + scale_msa) + shift_msa

        # Combined QKV attention
        qkv = self.attn["qkv"](norm_x)
        q, k, v = qkv.chunk(3, dim=-1)

        b, t = q.shape[0], q.shape[1]
        q = q.view(b, t, self.heads, -1).transpose(1, 2)  # [B, H, T, D]
        k = k.view(b, t, self.heads, -1).transpose(1, 2)  # [B, H, T, D]
        v = v.view(b, t, self.heads, -1).transpose(1, 2)  # [B, H, T, D]

        # Apply RoPE to Q and K (matching reference dit_block.py:95-96)
        if rotary_pos_emb is not None and position_ids is not None:
            # position_ids: [B, T] → index into freqs_cis [max_seq, D//2]
            # freqs for this forward: [B, T, D//2]
            freqs = rotary_pos_emb[position_ids]  # [B, T, D//2]
            q, k = _apply_rotary_emb(q, k, freqs)

        # Streaming KV cache (matching reference incremental_state)
        if incremental_state is not None:
            if "prev_k" in incremental_state:
                prev_k = incremental_state["prev_k"]  # [B, H, T_prev, D] (already has RoPE)
                prev_v = incremental_state["prev_v"]
                k = torch.cat([prev_k, k], dim=2)
                v = torch.cat([prev_v, v], dim=2)
            # Store current K/V for next chunk (after RoPE already applied)
            incremental_state["cur_k"] = k
            incremental_state["cur_v"] = v

            # Full attention mask: current chunk attends to all cached + all in chunk.
            # Reference uses full attention (no causal mask) within each chunk.
            total_len = k.shape[2]
            # [B, H, T, total_len]: prefix (cached) all zeros, chunk all zeros (full attention)
            attn_mask = torch.zeros(b, self.heads, t, total_len, device=q.device, dtype=q.dtype)

            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v)

        attn_out = attn_out.transpose(1, 2).reshape(b, t, self.inner_dim)
        attn_out = self.attn["proj"](attn_out)

        x = x + gate_msa * attn_out

        # Feed-forward
        ff_norm = self.ff_norm(x) * (1 + scale_mlp) + shift_mlp
        ff_out = self.mlp["fc2"](self.mlp_act(self.mlp["fc1"](ff_norm)))
        x = x + gate_mlp * ff_out

        return x


class KimiAudioFinalLayer(nn.Module):
    """Final layer matching checkpoint: final_layer.adaLN_modulation.1, final_layer.linear."""

    def __init__(self, dim: int, mel_dim: int):
        super().__init__()
        # Checkpoint: final_layer.adaLN_modulation.1.weight [4608, 2304] = [2*dim, dim]
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim),
        )
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, mel_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # c = [B, T, D], pass through adaLN for final scale/shift (matches original dit_block.py:196-200)
        emb = self.adaLN_modulation(c)  # [B, T, 2*dim]
        scale, shift = torch.chunk(emb, 2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return self.linear(x)


def _sinusoidal_embedding(t: torch.Tensor, dim: int = 256) -> torch.Tensor:
    """Generate sinusoidal timestep embedding [B, dim].

    Matches reference TimestepEmbedder.timestep_embedding exactly:
    freqs[i] = exp(-log(10000) * i / half)  # half = dim // 2
    output = [cos(t * freqs), sin(t * freqs)]
    """
    half_dim = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half_dim, device=t.device, dtype=torch.float32) / half_dim
    )
    args = t.unsqueeze(1).float() * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def _precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0,
                          interpolation_factor: float = 1.0) -> torch.Tensor:
    """Precompute RoPE frequency cis table [end, dim//2].

    Matches reference model.py:precompute_freqs_cis exactly.
    Returns complex tensor for rotary embedding via view_as_complex.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device).float()
    scale = 1.0 / interpolation_factor
    t *= scale
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def _apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor,
                      freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q and K tensors.

    Matches reference dit_block.py:apply_rotary_emb exactly.
    xq, xk: [B, H, T, D] — D must be even (head dim)
    freqs_cis: [B, T, D//2] complex tensor — already indexed by position_ids
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # freqs_cis shape: [B, T, D//2] → reshape to [B, 1, T, D//2] for head broadcast
    shape = [freqs_cis.shape[0], 1, freqs_cis.shape[1], freqs_cis.shape[2]]
    freqs_cis = freqs_cis.view(*shape)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)




class KimiAudioDetokenizer(nn.Module):
    """DiT-based audio detokenizer for Kimi-Audio TTS.

    Checkpoint key mapping:
        input_linear.weight/bias              [2304, 80] / [2304]
        semantic_token_embedding.weight       [16385, 2304]
        t_embedder.mlp.0.weight/bias          [2304, 256] / [2304]
        t_embedder.mlp.2.weight/bias          [2304, 2304] / [2304]
        blocks.{0-15}.attn.qkv.weight/bias    [6912, 2304] / [6912]
        blocks.{0-15}.attn.proj.weight/bias   [2304, 2304] / [2304]
        blocks.{0-15}.mlp.fc1.weight/bias     [9216, 2304] / [9216]
        blocks.{0-15}.mlp.fc2.weight/bias     [2304, 9216] / [2304]
        blocks.{0-15}.adaLN_modulation.1.*    [13824, 2304] / [13824]
        final_layer.adaLN_modulation.1.*      [4608, 2304] / [4608]
        final_layer.linear.weight/bias        [80, 2304] / [80]
    """

    def __init__(
        self,
        hidden_dim: int = 2304,
        num_heads: int = 18,
        dim_head: int = 128,
        ff_mult: int = 4,
        num_blocks: int = 16,
        mel_dim: int = 80,
        semantic_vocab_size: int = 16385,
        t_embed_dim: int = 256,
        normalize_mel: bool = False,
        mel_mean: float = -4.479605,
        mel_std: float = 3.4584913,
        max_seq_len: int = 4096,
        rope_base: float = 10000.0,
        rope_interpolation_factor: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.mel_dim = mel_dim
        self.t_embed_dim = t_embed_dim
        self.normalize_mel = normalize_mel
        self.mel_mean = mel_mean
        self.mel_std = mel_std
        self.max_seq_len = max_seq_len

        # Input projections
        self.input_linear = nn.Linear(mel_dim, hidden_dim)
        self.semantic_token_embedding = nn.Embedding(semantic_vocab_size, hidden_dim)

        # Timestep embedding: checkpoint uses t_embedder.mlp.{0,2}
        # t_embedder.mlp.0: Linear(256 → 2304), t_embedder.mlp.2: Linear(2304 → 2304)
        # Index 1 would be SiLU between them
        self.t_embedder = nn.ModuleDict(
            {
                "mlp": nn.Sequential(
                    nn.Linear(t_embed_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            }
        )

        # RoPE rotary position embeddings (matching reference config: use_rope=true)
        rope_dim = hidden_dim // num_heads
        self.register_buffer(
            "rotary_pos_emb",
            _precompute_freqs_cis(rope_dim, max_seq_len, rope_base, rope_interpolation_factor),
        )

        # DiT blocks
        self.blocks = nn.ModuleList(
            [KimiAudioDiTBlock(hidden_dim, num_heads, dim_head, ff_mult) for _ in range(num_blocks)]
        )

        # Output projection
        self.final_layer = KimiAudioFinalLayer(hidden_dim, mel_dim)

    def forward(
        self,
        x: torch.Tensor,
        codes: torch.Tensor,
        t: torch.Tensor,
        incremental_state: dict | None = None,
    ) -> torch.Tensor:
        """Forward pass through the DiT detokenizer.

        Args:
            x: noised mel input [B, T, 80]
            codes: audio code tokens [B, T]
            t: diffusion timestep [B] — will be scaled by 1000 before embedding
            incremental_state: optional KV cache dict for streaming inference

        Returns:
            velocity (mel derivative) [B, T, 80]
        """
        x = self.input_linear(x)
        code_embed = self.semantic_token_embedding(codes)

        # Timestep embedding with scaling (matching original ode_wrapper.py:217-220)
        # Original: t = (t * 1000).long() before passing to TimestepEmbedder
        t_emb = _sinusoidal_embedding(t * 1000.0, self.t_embed_dim)
        t_emb = self.t_embedder["mlp"](t_emb)

        # Condition fused with time embedding (matching original model.py:336-337)
        # c = t_embedder(t).unsqueeze(1) + condition → [B, T, D]
        c = t_emb.unsqueeze(1) + code_embed

        # Generate position_ids for RoPE.
        # For non-incremental: position_ids = [0, 1, 2, ..., T-1] per batch
        # For incremental: position_ids starts from cached position offset
        b, seq_len = codes.shape
        device = codes.device

        if incremental_state is not None:
            # Track global position offset for RoPE
            if "_position_offset" not in incremental_state:
                incremental_state["_position_offset"] = 0
            pos_offset = incremental_state["_position_offset"]
            position_ids = torch.arange(
                pos_offset, pos_offset + seq_len, device=device, dtype=torch.long
            ).unsqueeze(0).expand(b, -1)
            incremental_state["_position_offset"] = pos_offset + seq_len
        else:
            position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(b, -1)

        # Move rotary_pos_emb to same device as input if needed
        rotary_pos_emb = self.rotary_pos_emb.to(device)

        for i, block in enumerate(self.blocks):
            block_state = None
            if incremental_state is not None:
                if i not in incremental_state:
                    incremental_state[i] = {}
                if "attn_kvcache" not in incremental_state[i]:
                    incremental_state[i]["attn_kvcache"] = {}
                block_state = incremental_state[i]["attn_kvcache"]
            x = block(x, c, block_state, rotary_pos_emb, position_ids)

        return self.final_layer(x, c)

    def clear_incremental_state(self) -> dict:
        """Create fresh incremental state dict for streaming inference."""
        return {}

    @torch.no_grad()
    def infer_chunk(
        self,
        codes: torch.Tensor,
        x: torch.Tensor,
        incremental_state: dict,
        ode_steps: int = 30,
    ) -> torch.Tensor:
        """Run ODE integration on a single chunk with KV caching.

        Args:
            codes: audio code tokens [B, T_chunk]
            x: random noise initialized mel [B, T_chunk, 80]
            incremental_state: mutable KV cache dict (updated in place)
            ode_steps: number of ODE steps for this chunk

        Returns:
            denoised mel [B, T_chunk, 80]
        """
        b = codes.shape[0]
        device = next(self.parameters()).device
        dtype = torch.float32

        # Linear scheduler (matching StreamingFlowMatchingScheduler)
        sigma_min = 1e-4
        t_min = 0.0
        t_max = 1.0 - sigma_min
        t_span = torch.linspace(t_min, t_max, ode_steps + 1, device=device, dtype=dtype)

        for step in range(1, len(t_span)):
            t_val = t_span[step - 1]
            t = torch.full((b,), t_val.item(), device=device, dtype=dtype)
            vel = self.forward(x, codes, t, incremental_state)
            # Sync to catch CUDA errors at the right location
            if step == 1:
                torch.cuda.synchronize()
            dt = t_span[step] - t_span[step - 1]
            x = x + dt * vel

        return x

    @torch.no_grad()
    def generate(
        self,
        codes: torch.Tensor,
        n_steps: int = 150,
        cfg_scale: float = 4.0,
    ) -> torch.Tensor:
        """Run ODE-based generation (single forward, no CFG).

        Matches original streaming detokenizer which uses use_cfg=False
        (audio_detokenizer/config.yaml: use_cfg: false).

        Args:
            codes: audio code tokens [B, T]
            n_steps: number of ODE integration steps
            cfg_scale: unused (kept for API compatibility)

        Returns:
            mel spectrogram [B, 80, T]
        """
        b, seq_len = codes.shape
        device = next(self.parameters()).device
        dtype = torch.float32

        # Match reference detokenizer: repeat_interleave(4) before DiT
        codes = codes.repeat_interleave(4, dim=1)
        seq_len = codes.shape[1]

        # Initialize from random noise
        x = torch.randn(b, seq_len, self.mel_dim, device=device, dtype=dtype)

        # Linear scheduler with sigma_min (matching StreamingFlowMatchingScheduler)
        # t_max = 1 - sigma_min, uniform linear steps (no cosine schedule)
        sigma_min = 1e-4
        t_min = 0.0
        t_max = 1.0 - sigma_min
        t_span = torch.linspace(t_min, t_max, n_steps + 1, device=device, dtype=dtype)

        for step in range(1, len(t_span)):
            t_val = t_span[step - 1]
            t = torch.full((b,), t_val.item(), device=device, dtype=dtype)

            # Single forward (no CFG, no dual forward)
            vel = self.forward(x, codes, t)

            # Euler step with constant step size (matching scheduler.py:49-52)
            dt = t_span[step] - t_span[step - 1]
            x = x + dt * vel

        mel_out = x.transpose(1, 2)  # [B, 80, T]

        # Denormalize mel to match vocoder training distribution.
        if self.normalize_mel:
            mel_out = mel_out * self.mel_std + self.mel_mean

        return mel_out

    @torch.no_grad()
    def generate_streaming_with_overlap(
        self,
        codes: torch.Tensor,
        vocoder: KimiAudioVocoder,
        n_steps: int = 15,
        chunk_size: int = 30,
    ) -> torch.Tensor:
        """Run streaming chunked ODE generation with Hamming window overlap smoothing.

        Matches reference Kimi-Audio detokenizer exactly:
        - 30 tokens per chunk (matching reference training distribution)
        - 15 ODE steps per chunk (matching reference streaming default)
        - Hamming window overlap-add at waveform level between chunks

        Args:
            codes: audio code tokens [B, T]
            vocoder: BigVGAN vocoder instance for waveform decoding
            n_steps: ODE steps per chunk (default 15, matching reference)
            chunk_size: token count per chunk (default 30, matching reference)

        Returns:
            waveform [B, T_audio] at 24kHz
        """
        b, seq_len = codes.shape
        device = next(self.parameters()).device

        # Match reference detokenizer: repeat_interleave(4) before DiT.
        # Each audio code is repeated 4x, producing 4 mel frames per code.
        # The reference detokenizer does this at line 134 of detokenizer/__init__.py
        # with upsample_factor=4. This maps N codes -> 4N mel frames -> 4N*480 audio samples.
        codes = codes.repeat_interleave(4, dim=1)
        seq_len = codes.shape[1]

        if vocoder is None:
            # Fallback: mel-only generation
            return self._generate_streaming_mel(codes, n_steps=n_steps,
                                                first_chunk_size=chunk_size,
                                                chunk_size=chunk_size)

        incremental_state = self.clear_incremental_state()

        # Build chunk boundaries (uniform chunk_size, matching reference)
        chunk_starts = []
        chunk_ends = []
        pos = 0
        while pos < seq_len:
            end = min(pos + chunk_size, seq_len)
            chunk_starts.append(pos)
            chunk_ends.append(end)
            pos = end

        audio_parts: list[torch.Tensor] = []
        pre_wav: torch.Tensor | None = None  # second half of prev decoded waveform
        pre_mel: torch.Tensor | None = None  # second half of prev mel for overlap

        frame_size = 480  # vocoder upsampling ratio

        for ci, (cs, ce) in enumerate(zip(chunk_starts, chunk_ends)):
            chunk_codes = codes[:, cs:ce]  # [B, T_chunk]
            t_chunk = ce - cs

            # Fresh noise per chunk (matching reference)
            x_chunk = torch.randn(b, t_chunk, self.mel_dim, device=device, dtype=torch.float32)

            mel_out = self.infer_chunk(
                codes=chunk_codes,
                x=x_chunk,
                incremental_state=incremental_state,
                ode_steps=n_steps,
            )  # [B, T_chunk, 80]

            # Denormalize mel to match vocoder training distribution.
            # The reference detokenizer uses normalize_mel=true which applies:
            #   mel * mel_std + mel_mean
            # after the DiT ODE output. Without this, the vocoder receives
            # normalized mel (mean~0, std~1) instead of acoustic mel values.
            if self.normalize_mel:
                mel_out = mel_out * self.mel_std + self.mel_mean

            # Update incremental state
            for layer_state in incremental_state.values():
                if not isinstance(layer_state, dict):
                    continue
                if "cur_k" in layer_state.get("attn_kvcache", {}):
                    kvcache = layer_state["attn_kvcache"]
                    kvcache["prev_k"] = kvcache.pop("cur_k")
                    kvcache["prev_v"] = kvcache.pop("cur_v")

            is_final = (ci == len(chunk_starts) - 1)

            if pre_mel is None:
                # First chunk: vocode full, return first half, save second half
                concat_mel = mel_out  # [B, T_chunk, 80]
                wav = vocoder(concat_mel.transpose(1, 2))  # [B, T_audio]

                if is_final:
                    audio_parts.append(wav)
                else:
                    chunk_audio_len = t_chunk * frame_size
                    half = chunk_audio_len // 2
                    audio_parts.append(wav[:, :half])
                    pre_wav = wav[:, half:]
                    pre_mel = mel_out[:, -t_chunk // 2:, :]  # save last half mel
            else:
                # Subsequent chunks: vocode new mel only, then crossfade boundary
                # with pre_wav tail to smooth the transition.
                new_wav = vocoder(mel_out.transpose(1, 2))  # [B, T_chunk * 480]
                new_wav_len = new_wav.shape[-1]

                if is_final:
                    # For final chunk, simply concatenate pre_wav + new_wav.
                    # This preserves correct total length.
                    if pre_wav is not None and pre_wav.shape[-1] > 0:
                        final_audio = torch.cat([pre_wav, new_wav], dim=-1)
                    else:
                        final_audio = new_wav
                    audio_parts.append(final_audio)
                    pre_wav = None
                    pre_mel = None
                else:
                    # Crossfade pre_wav tail into new_wav head, output only new_wav length.
                    # blend_len = min(pre_wav_len, new_wav_len // 4)
                    if pre_wav is not None and pre_wav.shape[-1] > 0:
                        overlap_len = pre_wav.shape[-1]
                        blend_len = min(overlap_len, new_wav_len // 4)
                        if blend_len > 0:
                            hamming = torch.hamming_window(
                                2 * blend_len, dtype=new_wav.dtype, device=new_wav.device
                            ).unsqueeze(0)
                            blended_start = (
                                pre_wav[:, -blend_len:] * hamming[:, blend_len:]
                                + new_wav[:, :blend_len] * hamming[:, :blend_len]
                            )
                            chunk_audio = torch.cat([blended_start, new_wav[:, blend_len:]], dim=-1)
                        else:
                            chunk_audio = new_wav
                    else:
                        chunk_audio = new_wav

                    audio_parts.append(chunk_audio)

                    # Save last half of new audio as tail for next chunk
                    half_new = new_wav_len // 2
                    pre_wav = new_wav[:, half_new:]
                    pre_mel = mel_out[:, -t_chunk // 2:, :]

            if (ci + 1) % 5 == 0:
                torch.cuda.empty_cache()

        if pre_wav is not None and pre_wav.shape[-1] > 0:
            audio_parts.append(pre_wav)

        total_samples = sum(p.shape[-1] for p in audio_parts)
        logger.info(
            "Kimia-Audio generate_streaming_with_overlap: %d chunks, "
            "%d audio_parts, total_samples=%d, expected=%d",
            len(chunk_starts), len(audio_parts), total_samples,
            codes.shape[1] * frame_size,
        )

        return torch.cat(audio_parts, dim=-1)

    @torch.no_grad()
    def generate_streaming(
        self,
        codes: torch.Tensor,
        vocoder: KimiAudioVocoder | None = None,
        n_steps: int = 15,
        first_chunk_size: int = 30,
        chunk_size: int = 30,
    ) -> torch.Tensor:
        """Run streaming chunked ODE generation (matching reference implementation).

        Processes audio codes in chunks with independent noise initialization per chunk
        and KV caching across chunks. Each chunk gets its own ODE trajectory from fresh
        random noise, which is how the reference model was trained.

        Uses Hamming window overlap smoothing at the waveform level to eliminate
        chunk boundary artifacts (clicks, buzzing).

        Args:
            codes: audio code tokens [B, T]
            vocoder: BigVGAN vocoder instance (if None, returns mel without overlap)
            n_steps: ODE steps per chunk (default 15, matching reference)
            first_chunk_size: token count for first chunk (default 30, matching reference)
            chunk_size: token count for subsequent chunks (default 30, matching reference)

        Returns:
            waveform [B, T_audio] at 24kHz, or mel [B, 80, T] if vocoder is None
        """
        if vocoder is None:
            # Fallback: mel-only generation without overlap smoothing
            return self._generate_streaming_mel(
                codes, n_steps=n_steps,
                first_chunk_size=first_chunk_size,
                chunk_size=chunk_size,
            )
        return self.generate_streaming_with_overlap(
            codes, vocoder=vocoder, n_steps=n_steps, chunk_size=chunk_size,
        )

    def _generate_streaming_mel(
        self,
        codes: torch.Tensor,
        n_steps: int = 15,
        first_chunk_size: int = 30,
        chunk_size: int = 30,
    ) -> torch.Tensor:
        """Run streaming chunked ODE generation returning mel spectrogram (no vocoder).

        Legacy path for when vocoder is not available. Chunks are concatenated
        without overlap smoothing.

        Returns:
            mel spectrogram [B, 80, T]
        """
        b, seq_len = codes.shape
        device = next(self.parameters()).device
        dtype = torch.float32

        # Match reference detokenizer: repeat_interleave(4) before DiT
        codes = codes.repeat_interleave(4, dim=1)
        seq_len = codes.shape[1]

        incremental_state = self.clear_incremental_state()

        chunk_starts = []
        chunk_ends = []
        pos = 0
        first_end = min(first_chunk_size, seq_len)
        chunk_starts.append(0)
        chunk_ends.append(first_end)
        pos = first_end
        while pos < seq_len:
            end = min(pos + chunk_size, seq_len)
            chunk_starts.append(pos)
            chunk_ends.append(end)
            pos = end

        mel_parts = []

        for ci, (cs, ce) in enumerate(zip(chunk_starts, chunk_ends)):
            chunk_codes = codes[:, cs:ce]
            t_chunk = ce - cs

            x_chunk = torch.randn(b, t_chunk, self.mel_dim, device=device, dtype=dtype)

            mel_out = self.infer_chunk(
                codes=chunk_codes,
                x=x_chunk,
                incremental_state=incremental_state,
                ode_steps=n_steps,
            )

            for layer_state in incremental_state.values():
                if not isinstance(layer_state, dict):
                    continue
                if "cur_k" in layer_state.get("attn_kvcache", {}):
                    kvcache = layer_state["attn_kvcache"]
                    kvcache["prev_k"] = kvcache.pop("cur_k")
                    kvcache["prev_v"] = kvcache.pop("cur_v")

            mel_parts.append(mel_out)

            if (ci + 1) % 5 == 0:
                torch.cuda.empty_cache()

        mel = torch.cat(mel_parts, dim=1)

        # Denormalize mel to match vocoder training distribution.
        if self.normalize_mel:
            mel = mel * self.mel_std + self.mel_mean

        return mel.transpose(1, 2)


# ---------------------------------------------------------------------------
# HiFi-GAN / BigVGAN Vocoder Components
# ---------------------------------------------------------------------------


class KimiAudioSnakeBeta(nn.Module):
    """SnakeBeta activation matching reference Kimi-Audio implementation.

    Reference formula (with alpha_logscale=True):
        x + (1.0 / (exp(beta) + eps)) * sin(x * exp(alpha))^2

    Checkpoint keys: {prefix}.act.alpha, {prefix}.act.beta
    The checkpoint stores alpha/beta in log-space (snake_logscale=True).
    """

    def __init__(self, channels: int, alpha_logscale: bool = True):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        if alpha_logscale:
            # Log-scale: initialize to zeros (exp(0) = 1)
            self.register_parameter("alpha", nn.Parameter(torch.zeros(channels)))
            self.register_parameter("beta", nn.Parameter(torch.zeros(channels)))
        else:
            # Linear scale: initialize to ones
            self.register_parameter("alpha", nn.Parameter(torch.ones(channels)))
            self.register_parameter("beta", nn.Parameter(torch.ones(channels)))
        self.no_div_by_zero = 0.000000001

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        # Reference formula: x + (1/beta) * sin^2(x * alpha)
        return x + (1.0 / (beta + self.no_div_by_zero)) * torch.sin(x * alpha).pow(2)


class KimiAudioActivation1d(nn.Module):
    """Anti-aliased activation wrapper: upsample(2x) -> activation -> downsample(2x).

    Matches original Activation1d from alias_free_activation/torch/act.py.
    Each activation has learnable lowpass/upsample filters stored in checkpoint.

    Checkpoint keys per activation:
        {prefix}.upsample.filter                    [1, 1, 12]
        {prefix}.downsample.lowpass.filter          [1, 1, 12]
    """

    def __init__(self, channels: int, alpha_logscale: bool = True):
        super().__init__()
        self.act = KimiAudioSnakeBeta(channels, alpha_logscale=alpha_logscale)
        # Filter kernel size (must be even for alias-free design)
        self.kernel_size = 12
        self.ratio = 2
        # Compute padding matching original UpSample1d / LowPassFilter1d
        self.pad = self.kernel_size // self.ratio - 1
        self.pad_left = self.pad * self.ratio + (self.kernel_size - self.ratio) // 2
        self.pad_right = self.pad * self.ratio + (self.kernel_size - self.ratio + 1) // 2
        # LowPassFilter1d padding
        self.lp_pad_left = self.kernel_size // 2 - 1
        self.lp_pad_right = self.kernel_size // 2
        # Filter buffers (loaded from checkpoint during weight loading)
        self.register_buffer("upsample_filter", torch.zeros(1, 1, self.kernel_size))
        self.register_buffer("downsample_filter", torch.zeros(1, 1, self.kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._upsample(x)
        x = self.act(x)
        x = self._downsample(x)
        return x

    def _upsample(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample by 2x using conv_transpose1d with learnable filter."""
        _, c, _ = x.shape
        # Replicate padding
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        # conv_transpose1d with expanded filter
        f = self.upsample_filter.repeat(c, 1, 1)  # [c, 1, 12]
        x = self.ratio * F.conv_transpose1d(x, f, stride=self.ratio, groups=c)
        # Crop padding artifacts
        x = x[..., self.pad_left : -self.pad_right if self.pad_right > 0 else None]
        return x

    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        """Lowpass filter with stride=2 for decimation."""
        c = x.shape[1]
        # Replicate padding
        x = F.pad(x, (self.lp_pad_left, self.lp_pad_right), mode="replicate")
        f = self.downsample_filter.repeat(c, 1, 1)  # [c, 1, 12]
        x = F.conv1d(x, f, stride=self.ratio, groups=c)
        return x


class KimiAudioAMPBlock1(nn.Module):
    """AMPBlock1: BigVGAN residual block with dilated convs and anti-aliased activations.

    Architecture (matching original AMPBlock1):
        x -> [act -> conv1(dilation) -> act -> conv2(dilation=1)] x 3 -> + x
    Dilations: (1, 3, 5) for convs1, (1, 1, 1) for convs2
    Kernel sizes cycle through [3, 5, 7, 11] per block.
    6 anti-aliased activations per block.

    Checkpoint key structure:
        resblocks.{N}.convs1.{0-2}.weight_g/v/bias
        resblocks.{N}.convs2.{0-2}.weight_g/v/bias
        resblocks.{N}.activations.{0-5}.act.alpha/beta
        resblocks.{N}.activations.{0-5}.upsample.filter
        resblocks.{N}.activations.{0-5}.downsample.lowpass.filter
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple = (1, 3, 5),
                 alpha_logscale: bool = True):
        super().__init__()
        self.dilation = dilation

        # Convs1 with dilations (1, 3, 5)
        self.convs1 = nn.ModuleList()
        for d in dilation:
            padding = (kernel_size * d - d) // 2
            self.convs1.append(nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=padding))

        # Convs2 with dilation=1
        self.convs2 = nn.ModuleList()
        for _ in range(len(dilation)):
            padding = (kernel_size - 1) // 2
            self.convs2.append(nn.Conv1d(channels, channels, kernel_size, padding=padding))

        # 6 anti-aliased activations
        self.activations = nn.ModuleList(
            [KimiAudioActivation1d(channels, alpha_logscale=alpha_logscale)
             for _ in range(len(self.convs1) + len(self.convs2))]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual: each branch operates on original x, residuals accumulate.

        Matches reference AMPBlock1:
            for c1, c2, a1, a2 in zip(convs1, convs2, acts1, acts2):
                xt = a1(x); xt = c1(xt); xt = a2(xt); xt = c2(xt); x = xt + x
        """
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x


class KimiAudioVocoder(nn.Module):
    """BigVGAN vocoder for Kimi-Audio TTS.

    Architecture:
        conv_pre(80->2048) -> [ups->4*AMPBlock1] x 7 stages -> activation_post -> conv_post(16->1)
    Upsampling strides: [5, 2, 2, 2, 2, 3, 2] (total ratio = 480 = 24000/50)

    Matching original BigVGAN: resblocks are averaged per stage (xs / num_kernels).
    Uses snake_logscale=True to match checkpoint parameterization.
    """

    def __init__(self):
        super().__init__()
        # Channel progression: 2048 -> 1024 -> 512 -> 256 -> 128 -> 64 -> 32 -> 16
        self.channels = [2048, 1024, 512, 256, 128, 64, 32, 16]
        self.ups_kernels = [9, 4, 4, 4, 4, 5, 4]
        self.ups_strides = [5, 2, 2, 2, 2, 3, 2]
        self.num_resblocks_per_stage = 4
        self.num_stages = 7
        self.alpha_logscale = True  # Matches checkpoint config.json: snake_logscale=True

        # Initial conv
        self.conv_pre = nn.Conv1d(80, self.channels[0], kernel_size=7, padding=3)

        # Upsampling + residual blocks
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        # Kernel sizes cycle through [3, 5, 7, 11] per resblock
        kernel_cycle = [3, 5, 7, 11]
        for i in range(self.num_stages):
            ch_in = self.channels[i]
            ch_out = self.channels[i + 1]
            k = self.ups_kernels[i]
            s = self.ups_strides[i]
            padding = (k - s) // 2
            up_conv = nn.ConvTranspose1d(
                ch_in,
                ch_out,
                kernel_size=k,
                stride=s,
                padding=padding,
            )
            self.ups.append(nn.ModuleDict({"0": up_conv}))
            for j in range(self.num_resblocks_per_stage):
                rb_idx = i * self.num_resblocks_per_stage + j
                ks = kernel_cycle[rb_idx % 4]
                self.resblocks.append(KimiAudioAMPBlock1(ch_out, kernel_size=ks,
                                                         alpha_logscale=self.alpha_logscale))

        # Post-activation and final conv
        self.activation_post = KimiAudioActivation1d(self.channels[-1],
                                                     alpha_logscale=self.alpha_logscale)
        self.conv_post = nn.Conv1d(self.channels[-1], 1, kernel_size=7, padding=3)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrogram to waveform.

        Args:
            mel: [B, 80, T_mel]

        Returns:
            waveform: [B, T_audio]
        """
        x = self.conv_pre(mel)
        for i in range(self.num_stages):
            x = self.ups[i]["0"](x)
            # Average across resblocks (matching BigVGAN: xs / num_kernels)
            xs = None
            for j in range(self.num_resblocks_per_stage):
                idx = i * self.num_resblocks_per_stage + j
                if xs is None:
                    xs = self.resblocks[idx](x)
                else:
                    xs += self.resblocks[idx](x)
            x = xs / self.num_resblocks_per_stage
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.clamp(x, min=-1.0, max=1.0)
        return x.squeeze(1)


def _load_vocoder_weights(
    vocoder: KimiAudioVocoder,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Load vocoder weights from weight_norm (weight_g/weight_v) format.

    Applies weight_norm to Conv1d layers, loads the checkpoint's weight_g/weight_v,
    then removes weight_norm to freeze the reconstructed weights for inference.
    This ensures the weight_norm normalization is applied correctly.

    Also loads anti-aliasing filter parameters for each Activation1d module.
    Total checkpoint keys: 1206 (conv weights + filter params).
    """

    def load_conv_wn(module: nn.Conv1d, prefix: str) -> None:
        """Load weight_norm Conv1d from checkpoint weight_g/weight_v."""
        # Apply weight_norm to create weight_g and weight_v parameters
        wn_module = weight_norm(module)
        # Load checkpoint weight_g and weight_v
        wn_module.weight_g.data = state_dict[f"{prefix}.weight_g"].clone()
        wn_module.weight_v.data = state_dict[f"{prefix}.weight_v"].clone()
        if f"{prefix}.bias" in state_dict:
            module.bias.data = state_dict[f"{prefix}.bias"].clone()
        # Remove weight_norm to freeze the reconstructed weight
        remove_weight_norm(wn_module)

    def load_activation_filters(activation: KimiAudioActivation1d, prefix: str) -> None:
        """Load anti-aliasing filter parameters from checkpoint."""
        activation.upsample_filter.data = state_dict[f"{prefix}.upsample.filter"].clone()
        activation.downsample_filter.data = state_dict[f"{prefix}.downsample.lowpass.filter"].clone()

    # conv_pre
    load_conv_wn(vocoder.conv_pre, "conv_pre")

    # ups: checkpoint keys are ups.{i}.0.weight_g etc.
    for i in range(vocoder.num_stages):
        load_conv_wn(vocoder.ups[i]["0"], f"ups.{i}.0")

    # resblocks
    for i in range(vocoder.num_stages * vocoder.num_resblocks_per_stage):
        rb = vocoder.resblocks[i]
        for j in range(3):
            load_conv_wn(rb.convs1[j], f"resblocks.{i}.convs1.{j}")
            load_conv_wn(rb.convs2[j], f"resblocks.{i}.convs2.{j}")
        for j in range(6):
            act = rb.activations[j]
            act.act.alpha.data = state_dict[f"resblocks.{i}.activations.{j}.act.alpha"].clone()
            act.act.beta.data = state_dict[f"resblocks.{i}.activations.{j}.act.beta"].clone()
            load_activation_filters(act, f"resblocks.{i}.activations.{j}")

    # activation_post
    load_activation_filters(vocoder.activation_post, "activation_post")
    vocoder.activation_post.act.alpha.data = state_dict["activation_post.act.alpha"].clone()
    vocoder.activation_post.act.beta.data = state_dict["activation_post.act.beta"].clone()

    # conv_post
    load_conv_wn(vocoder.conv_post, "conv_post")


# ---------------------------------------------------------------------------
# Main Stage Model
# ---------------------------------------------------------------------------


class KimiaAudioCode2WavForConditionalGeneration(nn.Module):
    """Stage-1 code2wav model for Kimi-Audio TTS.

    Takes audio code token IDs from the AR stage and decodes them into
    waveforms via the DiT-based audio detokenizer and HiFi-GAN vocoder.

    Input: audio code token IDs (offset by kimia_token_offset=152064)
    Output: waveform tensors at 24kHz sample rate
    """

    input_modalities = "audio"

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False

        self._detokenizer: KimiAudioDetokenizer | None = None
        self._vocoder: KimiAudioVocoder | None = None
        self._reference_detokenizer: Any | None = None
        self._output_sample_rate: int = 24000
        self._token_offset: int = 152064
        self._audio_vocab_size: int = 16384

        # Use reference Kimi-Audio detokenizer by default. vLLM-native DiT is
        # deprecated due to CUDA instability. Set VLLM_OMNI_KIMIA_USE_REF_DETOKENIZER=0
        # to fall back to vLLM-native DiT (for debugging only).
        self._use_reference_detokenizer = os.environ.get(
            "VLLM_OMNI_KIMIA_USE_REF_DETOKENIZER", "1"
        ) not in ("0", "false", "False")

    def _ensure_detokenizer_loaded(self) -> None:
        """Lazily load DiT detokenizer and HiFi-GAN vocoder."""
        if self._use_reference_detokenizer:
            if self._reference_detokenizer is not None:
                return
        else:
            if self._detokenizer is not None:
                return

        device = self.vllm_config.device_config.device

        # --- Load reference detokenizer if requested ---
        if self._use_reference_detokenizer:
            self._load_reference_detokenizer(device)
            # Also load vocoder for the reference path
            self._ensure_vocoder_loaded(device)
            return

        # --- Load DiT detokenizer ---
        detokenizer_dir = os.path.join(self.model_path, "audio_detokenizer")
        model_pt_path = os.path.join(detokenizer_dir, "model.pt")

        if not os.path.exists(model_pt_path):
            raise FileNotFoundError(
                f"DiT detokenizer not found at {model_pt_path}. "
                f"Expected audio_detokenizer/model.pt under model path {self.model_path}."
            )

        logger.info("Loading Kimi-Audio DiT detokenizer from %s ...", model_pt_path)

        checkpoint = torch.load(model_pt_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Strip "speech_model." prefix
        stripped = {}
        for k, v in state_dict.items():
            if k.startswith("speech_model."):
                stripped[k[len("speech_model.") :]] = v
            else:
                stripped[k] = v

        # Remove resampler.kernel (not used in inference)
        stripped.pop("resampler.kernel", None)

        # Enable mel denormalization to match reference detokenizer.
        # The reference uses normalize_mel=true with these values from config.yaml:
        #   mel_mean: -4.479605
        #   mel_std: 3.4584913
        # Without denormalization, the vocoder receives out-of-distribution mel input.
        detokenizer = KimiAudioDetokenizer(
            normalize_mel=True,
            mel_mean=-4.479605,
            mel_std=3.4584913,
        )
        result = detokenizer.load_state_dict(stripped, strict=False)
        if result.missing_keys:
            logger.warning("Detokenizer missing keys: %s", result.missing_keys)
        if result.unexpected_keys:
            logger.warning("Detokenizer unexpected keys: %s", result.unexpected_keys)

        detokenizer.to(device).eval()
        self._detokenizer = detokenizer

        # --- Load HiFi-GAN vocoder ---
        self._ensure_vocoder_loaded(device)

        logger.info(
            "Kimia-Audio detokenizer loaded: vocab_size=%d, sample_rate=%d, blocks=%d",
            self._detokenizer.semantic_token_embedding.num_embeddings - 1,
            self._output_sample_rate,
            len(self._detokenizer.blocks),
        )

    def _load_reference_detokenizer(self, device: torch.device) -> None:
        """Load reference Kimi-Audio StreamingSemanticFMWrapper detokenizer."""
        _REFERENCE_PATH = "/root/learning/Kimi-Audio"
        if _REFERENCE_PATH not in sys.path:
            sys.path.insert(0, _REFERENCE_PATH)

        from kimia_infer.models.detokenizer.semantic_fm_prefix_streaming import (
            StreamingSemanticFMWrapper,
        )

        detokenizer_dir = os.path.join(self.model_path, "audio_detokenizer")
        config_path = os.path.join(detokenizer_dir, "config.yaml")
        ckpt_path = os.path.join(detokenizer_dir, "model.pt")

        logger.info(
            "Loading reference detokenizer from %s (config=%s)",
            self.model_path,
            config_path,
        )

        self._reference_detokenizer = StreamingSemanticFMWrapper.from_pretrained(
            model_config=config_path,
            ckpt_path=ckpt_path,
            device=device,
            max_prompt_chunk=2,
            max_kv_cache_tokens=900,
            use_cfg=False,
            use_cfg_rescale=False,
            cfg_init=1.0,
            cfg_scale=4.0,
            cfg_schedule="linear",
        )
        logger.info("Reference detokenizer loaded successfully")

    def _ensure_vocoder_loaded(self, device: torch.device) -> None:
        """Load HiFi-GAN vocoder if not already loaded."""
        if self._vocoder is not None:
            return

        vocoder_dir = os.path.join(self.model_path, "vocoder")
        vocoder_path = os.path.join(vocoder_dir, "model.pt")

        if os.path.exists(vocoder_path):
            logger.info("Loading Kimi-Audio HiFi-GAN vocoder from %s ...", vocoder_path)
            vocoder_ckpt = torch.load(vocoder_path, map_location=device, weights_only=False)
            if isinstance(vocoder_ckpt, dict) and "generator" in vocoder_ckpt:
                vocoder_state = vocoder_ckpt["generator"]
            else:
                vocoder_state = vocoder_ckpt

            vocoder = KimiAudioVocoder()
            _load_vocoder_weights(vocoder, vocoder_state)
            vocoder.to(device).eval()
            self._vocoder = vocoder
        else:
            logger.warning("Vocoder not found at %s, waveform generation disabled", vocoder_path)
            self._vocoder = None

        if self._detokenizer is not None:
            logger.info(
                "Kimia-Audio detokenizer loaded: vocab_size=%d, sample_rate=%d, blocks=%d",
                self._detokenizer.semantic_token_embedding.num_embeddings - 1,
                self._output_sample_rate,
                len(self._detokenizer.blocks),
            )
        elif self._reference_detokenizer is not None:
            logger.info(
                "Kimia-Audio reference detokenizer loaded: sample_rate=%d",
                self._output_sample_rate,
            )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        **_: Any,
    ) -> torch.Tensor:
        """Return dummy embeddings for vLLM runner compatibility."""
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros(
            (input_ids.shape[0], 1),
            device=input_ids.device,
            dtype=torch.float32,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> None:
        """No logits computation needed for code2wav stage."""
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
        """Decode audio codes into waveform."""
        self._ensure_detokenizer_loaded()

        sr_tensor = torch.tensor(self._output_sample_rate, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        # Flatten and filter valid codes.
        # Input codes come from the AR stage with token_offset added (152064-168447 range).
        # We need to subtract the offset to get the actual audio codes [0, 16383].
        # During warmup (graph capture), input_ids are small dummy values.
        # Use value range check to distinguish warmup from real inference.
        num_codes = input_ids.shape[0]
        is_warmup = num_codes <= 2 and input_ids.max() < 10

        if is_warmup:
            # Return dummy waveform immediately — avoid any .cpu() or .item() during capture
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        codes = input_ids.reshape(-1).to(dtype=torch.long)
        # Audio codes may come from the AR stage in either format:
        # 1. With offset: [152064, 168447] (full token IDs)
        # 2. Without offset: [0, 16383] (audio code indices from audio_logits sampling)
        # Detect which format by checking if values are in the audio vocab range.
        if codes.min() >= self._token_offset:
            codes = codes - self._token_offset
        codes = codes.cpu()
        valid_mask = (codes >= 0) & (codes < self._audio_vocab_size)
        valid_codes = codes[valid_mask]

        logger.info(
            "Kimia-Audio code2wav: input %d codes, %d valid after filtering (range [0, %d))",
            codes.numel(),
            valid_codes.numel(),
            self._audio_vocab_size,
        )
        if valid_codes.numel() > 0:
            logger.info(
                "Kimia-Audio code2wav: valid codes range [%d, %d]",
                valid_codes.min().item(),
                valid_codes.max().item(),
            )

        if valid_codes.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        # Detect warmup inputs (all-zero codes) and skip heavy DiT/vocoder
        if valid_codes.numel() > 1 and valid_codes.max().item() == 0:
            logger.debug("Kimia-Audio code2wav: warmup detected (all-zero codes), returning dummy")
            # Return a short dummy waveform for warmup handshake
            dummy_len = self._output_sample_rate  # 1 second of silence
            dummy_waveform = torch.zeros(dummy_len, dtype=torch.float32)
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [dummy_waveform], "sr": [sr_tensor]},
            )

        device = self.vllm_config.device_config.device
        audio_codes = valid_codes.unsqueeze(0).to(device)
        _, t_len = audio_codes.shape

        logger.info(
            "Kimia-Audio code2wav: audio_codes shape=%s, min=%d, max=%d, unique=%d",
            list(audio_codes.shape), audio_codes.min().item(),
            audio_codes.max().item(), audio_codes.unique().numel(),
        )

        # Step 1+2: DiT generates mel + vocoder decodes to waveform with overlap smoothing
        if self._use_reference_detokenizer:
            # Reference detokenizer uses 30 ODE steps per chunk, 30 tokens per chunk
            logger.info("Kimia-Audio code2wav: using REFERENCE detokenizer (30 ODE steps/chunk, chunk_size=30)")

            # Match reference detokenize_streaming: repeat_interleave(4) before DiT.
            # The reference detokenizer/__init__.py line 134 does this with upsample_factor=4.
            # Each audio code -> 4 mel frames -> 4*480=1920 audio samples.
            audio_codes = audio_codes.repeat_interleave(4, dim=1)

            self._reference_detokenizer.clear_all_states()
            mel = self._reference_detokenizer.infer_mel(
                semantic_tokens=audio_codes.squeeze(0),
                ode_steps=30,
                chunk_size=30,
            )
            # mel: [T, 80] -> [B, 80, T], convert to float32 for vocoder
            mel = mel.transpose(0, 1).unsqueeze(0).float()  # [T,80] -> [80,T] -> [B,80,T]
            waveform = self._vocoder(mel)
        else:
            # vLLM-native DiT detokenizer — DEPRECATED due to CUDA instability
            logger.warning(
                "vLLM-native DiT detokenizer is deprecated due to CUDA instability. "
                "Use the reference detokenizer (default) instead. "
                "Set VLLM_OMNI_KIMIA_USE_REF_DETOKENIZER=1 (default) for reference."
            )
            logger.info("Kimia-Audio code2wav: streaming detokenization with %d ODE steps/chunk, chunk_size=30", 30)
            waveform = self._detokenizer.generate_streaming(
                audio_codes, vocoder=self._vocoder, n_steps=30, chunk_size=30,
            )

        # generate_streaming returns waveform [B, T_audio] when vocoder is available,
        # or mel [B, 80, T] when vocoder is not available (fallback)
        if waveform.dim() == 3:
            # Fallback: vocoder not available, mel already denormalized by detokenizer.
            logger.warning("No vocoder loaded, returning empty waveform")
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        logger.info(
            "Kimia-Audio vocoder output: shape=%s, min=%.4f, max=%.4f, mean=%.6f, std=%.6f",
            list(waveform.shape), waveform.min().item(), waveform.max().item(),
            waveform.mean().item(), waveform.std().item(),
        )

        # DC removal: center the waveform to match the reference model's
        # near-zero mean output. The vocoder inherently produces waveforms
        # with slight DC bias from the mel-to-waveform conversion. The
        # reference model's generation loop naturally produces centered audio
        # codes, but vLLM's prefill-path codes can introduce a DC component.
        # Removing it BEFORE peak normalization prevents amplifying the bias.
        dc_offset = waveform.mean().item()
        if abs(dc_offset) > 1e-4:
            waveform = waveform - dc_offset
            logger.info(
                "Kimia-Audio DC removal: mean=%.6f -> %.6f (removed %.6f)",
                dc_offset, waveform.mean().item(), dc_offset,
            )

        # Peak normalization to match reference Kimi-Audio output level.
        # The reference pipeline uses librosa.util.normalize(audio) * 0.95,
        # which scales the waveform so its peak amplitude is 0.95.
        # Without this, the vocoder output is much quieter than reference.
        peak = waveform.abs().max().item()
        _target_peak = 0.95
        _min_peak = 1e-6  # floor to prevent division by zero on silence
        if peak > _min_peak:
            _scale_factor = _target_peak / peak
            waveform = waveform * _scale_factor
            logger.info(
                "Kimia-Audio peak normalization: %.4f -> %.4f (scale=%.4f)",
                peak, _target_peak, _scale_factor,
            )
        else:
            logger.info(
                "Kimia-Audio skipping peak normalization (peak=%.6f below floor %.4f)",
                peak, _min_peak,
            )

        # Resample from vocoder native 24kHz to output sample rate
        native_sr = 24000
        if self._output_sample_rate != native_sr:
            import torchaudio.transforms as T
            resampler = T.Resample(
                orig_freq=native_sr, new_freq=self._output_sample_rate,
                dtype=waveform.dtype,
            ).to(waveform.device)
            waveform = resampler(waveform)
            logger.info(
                "Kimia-Audio resampled: %d Hz -> %d Hz, shape=%s",
                native_sr, self._output_sample_rate, list(waveform.shape),
            )

        # Estimate mel frame count from audio length
        mel_frame_count = t_len  # codes count before 4x upsample
        logger.info(
            "Kimia-Audio detokenizer: %d codes -> ~%d mel frames -> %d audio samples",
            t_len,
            mel_frame_count,
            waveform.shape[-1],
        )

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "audio": list(waveform.unbind(0)),
                "sr": [sr_tensor] * waveform.shape[0],
            },
        )

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput:
        """Wrap raw model outputs into OmniOutput format."""
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        # Handle deconstructed OmniOutput from CUDA graph replay.
        # vLLM's weak_ref_tensors converts NamedTuple to plain tuple during
        # graph capture. On replay, we get a plain 4-tuple:
        # (text_hidden_states, multimodal_outputs_dict, intermediate_tensors, next_token_id)
        if isinstance(model_outputs, (tuple, list)) and len(model_outputs) == 4:
            text_hs = model_outputs[0]
            mm_out = model_outputs[1]
            if isinstance(mm_out, dict):
                return OmniOutput(
                    text_hidden_states=text_hs,
                    multimodal_outputs=mm_out,
                )

        if isinstance(model_outputs, torch.Tensor):
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": model_outputs},
            )

        # Handle other tuple/list outputs (e.g., from wrapped forward calls)
        if isinstance(model_outputs, (tuple, list)):
            # If it contains OmniOutput, return it directly
            for item in model_outputs:
                if isinstance(item, OmniOutput):
                    return item
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [model_outputs]},
            )

        raise TypeError(
            f"KimiaAudioCode2Wav expected tensor or OmniOutput, got {type(model_outputs)}",
        )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """No weight loading here -- detokenizer loads from model.pt lazily."""
        return set()
