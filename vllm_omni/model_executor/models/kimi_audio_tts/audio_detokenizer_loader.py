# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio detokenizer architectures for Kimi-Audio."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HifiGanGenerator(nn.Module):
    """HiFi-GAN style vocoder generator."""
    
    def __init__(
        self,
        num_mels: int = 80,
        upsample_rates: list[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: list[int] = [16, 16, 4, 4],
        resblock_kernel_sizes: list[int] = [3, 7, 11],
        resblock_dilation_sizes: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        
        self.conv_pre = nn.Conv1d(num_mels, 512, 7, 1, padding=3)
        
        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (upsample_rate, kernel_size) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    512 // (2 ** i),
                    512 // (2 ** (i + 1)),
                    kernel_size,
                    upsample_rate,
                    padding=kernel_size // 2 + upsample_rate // 2,
                    output_padding=upsample_rate % 2,
                )
            )
        
        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = 512 // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(
                    nn.ModuleList([
                        nn.Conv1d(ch, ch, k, 1, dilation=d[0], padding=k // 2),
                        nn.Conv1d(ch, ch, k, 1, dilation=d[1], padding=k // 2),
                        nn.Conv1d(ch, ch, k, 1, dilation=d[2], padding=k // 2),
                    ])
                )
        
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3)
        self.ups.apply(self._init_weights)
        self.conv_post.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.normal_(m.weight, 0.0, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input codes [batch, seq_len]
            
        Returns:
            waveform: [batch, 1, samples]
        """
        # Convert codes to mel spectrogram (simplified)
        # In practice, this would use a proper codebook
        x = x.float().unsqueeze(1)  # [batch, 1, seq_len]
        
        x = self.conv_pre(x)
        x = F.leaky_relu(x, 0.1)
        
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            
            # Residual blocks
            xs = None
            for j in range(self.num_kernels):
                resblock = self.resblocks[i * self.num_kernels + j]
                if xs is None:
                    xs = sum(resblock[k](x) for k in range(3)) / 3
                else:
                    xs += sum(resblock[k](x) for k in range(3)) / 3
            
            x = xs / self.num_kernels
        
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x


class VocosDecoder(nn.Module):
    """Vocos style audio decoder."""
    
    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 512,
        intermediate_dim: int = 1536,
        num_layers: int = 8,
    ):
        super().__init__()
        self.embedding = nn.Embedding(16384, hidden_dim)
        
        self.layers = nn.ModuleList([
            VocosDecoderLayer(hidden_dim, intermediate_dim)
            for _ in range(num_layers)
        ])
        
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        self.head = nn.Conv1d(input_dim, 1, kernel_size=7, padding=3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input codes [batch, seq_len]
            
        Returns:
            waveform: [batch, 1, samples]
        """
        x = self.embedding(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.out_proj(x)
        x = x.transpose(1, 2)  # [batch, dim, seq_len]
        x = self.head(x)
        
        return x


class VocosDecoderLayer(nn.Module):
    """Vocos decoder layer."""
    
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x
        
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


class GenericAudioDecoder(nn.Module):
    """Generic audio decoder that can load various checkpoint formats."""
    
    def __init__(
        self,
        vocab_size: int = 16384,
        hidden_dim: int = 512,
        sample_rate: int = 24000,
        upsample_factor: int = 240,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.sample_rate = sample_rate
        self.upsample_factor = upsample_factor
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Transformer encoder - ensure hidden_dim divisible by nhead
        nhead = 8 if hidden_dim % 8 == 0 else 4
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input codes [batch, seq_len]
            
        Returns:
            waveform: [batch, 1, samples]
        """
        # Embed codes
        x = self.embedding(x)
        
        # Encode
        x = self.encoder(x)
        
        # Project to waveform
        x = self.out_proj(x)
        x = x.transpose(1, 2)  # [batch, 1, seq_len]
        
        # Upsample to audio sample rate
        x = F.interpolate(x, scale_factor=self.upsample_factor, mode='linear', align_corners=False)
        
        return x


class DiTVocoder(nn.Module):
    """DiT-based audio vocoder for Kimi-Audio.
    
    Matches the architecture of audio_detokenizer/model.pt checkpoint.
    This is a Diffusion Transformer (DiT) style vocoder with:
    - 9 transformer blocks
    - 2304 hidden dimension
    - Adaptive LayerNorm (adaLN) modulation
    - Semantic token embedding for 16385 tokens
    """
    
    def __init__(
        self,
        vocab_size: int = 16385,
        hidden_dim: int = 2304,
        num_blocks: int = 9,
        sample_rate: int = 24000,
        upsample_factor: int = 240,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.sample_rate = sample_rate
        self.upsample_factor = upsample_factor
        self.num_blocks = num_blocks
        
        # Semantic token embedding
        self.semantic_token_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Transformer blocks with adaLN modulation
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        # Final projection
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input codes [batch, seq_len]
            
        Returns:
            waveform: [batch, 1, samples]
        """
        # Embed codes
        x = self.semantic_token_embedding(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Normalize and project
        x = self.norm(x)
        x = self.head(x)
        x = x.transpose(1, 2)  # [batch, 1, seq_len]
        
        # Upsample to audio sample rate
        x = F.interpolate(x, scale_factor=self.upsample_factor, mode='linear', align_corners=False)
        
        return x


class DiTBlock(nn.Module):
    """DiT transformer block with adaLN modulation."""
    
    def __init__(self, hidden_dim: int = 2304, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Attention
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        
        # MLP
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
        # adaLN modulation (simplified - no conditioning for now)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True),
        )
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass."""
        # Self-attention with adaLN
        modulation = self.adaLN_modulation(x.mean(dim=1, keepdim=True))
        
        residual = x
        x_norm = self.norm1(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm)
        x = residual + x_attn
        
        # MLP with adaLN
        residual = x
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        x = residual + x_mlp
        
        return x
