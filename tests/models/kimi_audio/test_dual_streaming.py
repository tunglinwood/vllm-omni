# Copyright 2025 vLLM-Omni Team
"""Unit tests for Kimi Audio dual streaming implementation."""

import pytest
import torch
from unittest.mock import MagicMock, Mock


@pytest.fixture
def mock_model():
    """Create a mock KimiAudioLLMForConditionalGeneration model."""
    from vllm_omni.model_executor.models.kimi_audio.kimi_audio_llm import (
        KimiAudioLLMForConditionalGeneration,
    )
    from vllm.config import VllmConfig

    # Create minimal mock config
    mock_config = MagicMock()
    mock_config.hidden_size = 896
    mock_config.vocab_size = 168448
    mock_config.kimia_mimo_layers = 6
    mock_config.rms_norm_eps = 1e-5

    mock_vllm_config = MagicMock(spec=VllmConfig)
    mock_vllm_config.model_config = MagicMock()
    mock_vllm_config.model_config.hf_config = mock_config
    mock_vllm_config.model_config.dtype = torch.float32
    mock_vllm_config.cache_config = MagicMock()
    mock_vllm_config.quant_config = None

    # Mock the init_vllm_registered_model and other dependencies
    with pytest.MonkeyPatch.context() as m:
        # Mock all the heavy dependencies
        m.setattr(
            "vllm_omni.model_executor.models.kimi_audio.kimi_audio_llm.init_vllm_registered_model",
            MagicMock()
        )
        m.setattr(
            "vllm_omni.model_executor.models.kimi_audio.kimi_audio_llm.KimiAudioWhisperEncoder",
            MagicMock()
        )
        m.setattr(
            "vllm_omni.model_executor.models.kimi_audio.kimi_audio_llm.KimiAudioMultiModalProjector",
            MagicMock()
        )
        m.setattr(
            "vllm_omni.model_executor.models.kimi_audio.kimi_audio_llm.Qwen2DecoderLayer",
            MagicMock()
        )
        m.setattr(
            "vllm_omni.model_executor.models.kimi_audio.kimi_audio_llm.RMSNorm",
            MagicMock()
        )
        m.setattr(
            "vllm_omni.model_executor.models.kimi_audio.kimi_audio_llm.ColumnParallelLinear",
            MagicMock()
        )
        m.setattr(
            "vllm_omni.model_executor.models.kimi_audio.kimi_audio_llm.LogitsProcessor",
            MagicMock()
        )

        model = KimiAudioLLMForConditionalGeneration(
            vllm_config=mock_vllm_config,
            prefix=""
        )

    return model


@pytest.mark.asyncio
async def test_extension_point_flags(mock_model):
    """Test that extension point flags are set correctly."""
    assert getattr(mock_model, 'prefer_model_sampler') == True
    assert getattr(mock_model, 'has_preprocess') == True
    assert getattr(mock_model, 'has_postprocess') == True
    assert getattr(mock_model, 'have_multimodal_outputs') == True
    assert getattr(mock_model, 'postprocess_uses_hidden_states') == True
    assert getattr(mock_model, 'postprocess_uses_multimodal_outputs') == True
    assert getattr(mock_model, 'postprocess_uses_req_infos') == True


@pytest.mark.asyncio
async def test_state_tracking_initialization(mock_model):
    """Test that state tracking variables are initialized correctly."""
    assert mock_model._pending_audio_token is None
    assert mock_model._pending_audio_logits is None
    assert mock_model._generation_step == 0
    assert mock_model._audio_delay == 6
    assert mock_model._blank_token_id == 18
    assert mock_model._text_eos_id == 19
    assert mock_model._token_offset == 152064


@pytest.mark.asyncio
async def test_dual_stream_embedding_no_audio(mock_model):
    """Test dual-stream embedding when no audio token is pending."""
    # Setup
    input_ids = torch.tensor([[100, 200, 300]])
    mock_model._pending_audio_token = None

    # Mock the embed_tokens method
    mock_embed = MagicMock(return_value=torch.randn(1, 3, 896))
    mock_model.model.model.embed_tokens = mock_embed

    # Execute
    inputs_embeds = mock_model.embed_input_ids(input_ids)

    # Verify
    assert inputs_embeds.shape == (1, 3, 896)
    mock_embed.assert_called_once_with(input_ids)


@pytest.mark.asyncio
async def test_dual_stream_embedding_with_audio(mock_model):
    """Test dual-stream embedding with audio token fusion."""
    # Setup
    input_ids = torch.tensor([[100, 200, 300]])
    audio_token = torch.tensor([[152064, 152065, 152066]])
    mock_model._pending_audio_token = audio_token

    # Mock the embed_tokens method to return different embeddings
    def mock_embed(tokens):
        if torch.equal(tokens, input_ids):
            return torch.ones(1, 3, 896)  # Text embeddings
        elif torch.equal(tokens, audio_token):
            return torch.ones(1, 3, 896) * 2  # Audio embeddings
        return torch.randn(tokens.shape[0], tokens.shape[1], 896)

    mock_model.model.model.embed_tokens = MagicMock(side_effect=mock_embed)

    # Execute
    inputs_embeds = mock_model.embed_input_ids(input_ids)

    # Verify - should be text_emb + audio_emb = 1 + 2 = 3
    assert inputs_embeds.shape == (1, 3, 896)
    assert torch.allclose(inputs_embeds, torch.ones(1, 3, 896) * 3)


@pytest.mark.asyncio
async def test_audio_delay_first_6_steps(mock_model):
    """Test that first 6 audio tokens are BLANK."""
    # Setup
    mock_model._pending_audio_logits = torch.randn(1, 168448)
    logits = torch.randn(1, 168448)
    sampling_metadata = MagicMock()

    # Mock the sampler
    mock_sampler_output = MagicMock()
    mock_sampler_output.sampled_token_ids = torch.tensor([[100]])
    mock_model._stock_sampler = MagicMock(return_value=mock_sampler_output)

    # Test first 6 steps
    for step in range(6):
        mock_model._generation_step = step
        result = mock_model.sample(logits, sampling_metadata)

        # Verify audio token is BLANK (18)
        assert mock_model._pending_audio_token is not None
        assert mock_model._pending_audio_token.item() == 18, f"Step {step}: expected BLANK (18), got {mock_model._pending_audio_token.item()}"


@pytest.mark.asyncio
async def test_audio_token_range_after_delay(mock_model):
    """Test that audio tokens are in correct range after delay."""
    # Setup
    mock_model._generation_step = 10  # Past delay
    mock_model._pending_audio_logits = torch.randn(1, 168448)
    logits = torch.randn(1, 168448)
    sampling_metadata = MagicMock()

    # Mock the sampler
    mock_sampler_output = MagicMock()
    mock_sampler_output.sampled_token_ids = torch.tensor([[100]])
    mock_model._stock_sampler = MagicMock(return_value=mock_sampler_output)

    # Execute
    result = mock_model.sample(logits, sampling_metadata)

    # Verify audio token is in range [152064, 168447]
    assert mock_model._pending_audio_token is not None
    audio_token = mock_model._pending_audio_token.item()
    assert 152064 <= audio_token <= 168447, f"Audio token {audio_token} not in range [152064, 168447]"


@pytest.mark.asyncio
async def test_generation_step_increments(mock_model):
    """Test that generation step increments correctly."""
    # Setup
    mock_model._generation_step = 5
    mock_model._pending_audio_logits = torch.randn(1, 168448)
    logits = torch.randn(1, 168448)
    sampling_metadata = MagicMock()

    # Mock the sampler
    mock_sampler_output = MagicMock()
    mock_sampler_output.sampled_token_ids = torch.tensor([[100]])
    mock_model._stock_sampler = MagicMock(return_value=mock_sampler_output)

    # Execute
    initial_step = mock_model._generation_step
    mock_model.sample(logits, sampling_metadata)

    # Verify
    assert mock_model._generation_step == initial_step + 1


@pytest.mark.asyncio
async def test_state_reset_on_requests_finished(mock_model):
    """Test that state is reset after request completion."""
    # Set some state
    mock_model._pending_audio_token = torch.tensor([152064])
    mock_model._pending_audio_logits = torch.randn(1, 168448)
    mock_model._generation_step = 10

    # Finish request
    mock_model.on_requests_finished(["test_req_id"])

    # Verify reset
    assert mock_model._pending_audio_token is None
    assert mock_model._pending_audio_logits is None
    assert mock_model._generation_step == 0


@pytest.mark.asyncio
async def test_make_omni_output(mock_model):
    """Test make_omni_output packages output correctly."""
    # Setup
    text_hidden = torch.randn(1, 896)
    audio_logits = torch.randn(1, 168448)
    audio_token = torch.tensor([[152064]])
    mock_model._pending_audio_token = audio_token

    model_outputs = {
        "text_hidden_states": text_hidden,
        "audio_logits": audio_logits,
    }

    # Execute
    output = mock_model.make_omni_output(model_outputs)

    # Verify
    assert output.text_hidden_states is text_hidden
    assert "audio_logits" in output.multimodal_outputs
    assert "audio_tokens" in output.multimodal_outputs
    assert output.next_token_id is audio_token


@pytest.mark.asyncio
async def test_postprocess(mock_model):
    """Test postprocess returns correct state dict."""
    # Setup
    mock_model._pending_audio_token = torch.tensor([152064])
    mock_model._generation_step = 10
    hidden_states = torch.randn(1, 896)
    multimodal_outputs = {"audio_logits": torch.randn(1, 168448)}

    # Execute
    result = mock_model.postprocess(hidden_states, multimodal_outputs)

    # Verify
    assert "audio_token" in result
    assert "generation_step" in result
    assert result["generation_step"] == 10
