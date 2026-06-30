# Kimi Audio Dual-Stream Fix - Summary

## Problem
The Kimi Audio model generates EOS immediately (empty text output) because it receives BLANK tokens (151666) instead of actual audio tokens (152064-168447).

## Root Cause
The reference Kimi-Audio implementation uses a dual-stream architecture:
- **Audio stream**: Contains actual audio tokens from Glm4Tokenizer
- **Text stream**: Contains text tokens with BLANK placeholders

However, vllm-omni uses a single-stream architecture:
- **Single stream**: Contains text tokens with BLANK placeholders
- Whisper continuous features are fused into embeddings

The model was trained to expect actual audio tokens, but vllm-omni provides BLANK tokens, causing the model to generate EOS immediately.

## Solution
Intercept audio processing in the API server and tokenize audio before sending to the model:

### 1. API Server Changes (`serving_chat.py`)

#### Added Kimi Audio Detection
```python
def _is_kimi_audio_model(self) -> bool:
    """Check if the current model is Kimi Audio."""
    model_arch = getattr(self.model_config.hf_config, "architectures", [])
    if isinstance(model_arch, list) and len(model_arch) > 0:
        arch_name = model_arch[0] if isinstance(model_arch[0], str) else str(model_arch[0])
        return "KimiAudio" in arch_name
    return False
```

#### Added Audio Preparation Handler
```python
async def _prepare_kimi_audio_inputs(
    self,
    messages: list[ChatCompletionMessageParam],
    request: ChatLikeRequest | ResponsesRequest,
) -> tuple[list[ChatCompletionMessageParam], dict[str, Any] | None]:
    """Extract and tokenize audio for Kimi Audio models."""
    # Extract audio from messages (supports multiple formats)
    # Materialize audio (decode base64, load from URL, etc.)
    # Tokenize using Glm4Tokenizer
    # Store tokens in class variable for model to access
```

#### Modified Preprocessing Flow
```python
async def _preprocess_chat(...):
    deferred_multi_modal_data: dict[str, Any] | None = None
    if self._needs_multistage_multimodal_split():
        messages, deferred_multi_modal_data = await self._prepare_multistage_multimodal_inputs(
            messages, request,
        )
    elif self._is_kimi_audio_model():
        # For Kimi Audio, always tokenize audio in the API server
        messages, deferred_multi_modal_data = await self._prepare_kimi_audio_inputs(
            messages, request,
        )
```

### 2. Audio Tokenization

#### Load Glm4Tokenizer
```python
from transformers import AutoTokenizer

self._audio_tokenizer = AutoTokenizer.from_pretrained(
    "THUDM/glm-4-voice-tokenizer",
    trust_remote_code=True
)
```

#### Tokenize Audio
```python
# Tokenize audio file
audio_tokens = self._audio_tokenizer.tokenize(audio_path=tmp_path)
# Add offset to get actual token IDs
audio_tokens = audio_tokens + 152064  # kimia_token_offset
# Convert to list
audio_tokens_list = audio_tokens.squeeze(0).cpu().numpy().tolist()
```

#### Store for Model Access
```python
from vllm_omni.model_executor.models.kimi_audio.kimi_audio_llm import KimiAudioLLMForConditionalGeneration
KimiAudioLLMForConditionalGeneration._pending_audio_tokens = audio_tokens_list
```

### 3. Model-Side Token Replacement (Already Implemented)

In `kimi_audio_llm.py` forward():
```python
# Check for pending audio tokens from serving_chat.py
if hasattr(self, "_pending_audio_tokens") and self._pending_audio_tokens is not None:
    self._current_audio_tokens = self._pending_audio_tokens
    self._pending_audio_tokens = None  # Clear after use

# Replace BLANK tokens in input_ids with actual audio tokens
if self._current_audio_tokens is not None and input_ids is not None:
    blank_mask = (input_ids == self._blank_token_id)
    if len(self._current_audio_tokens) == blank_mask.sum().item():
        audio_tokens_tensor = torch.tensor(
            self._current_audio_tokens,
            dtype=input_ids.dtype,
            device=input_ids.device
        )
        input_ids = input_ids.clone()
        input_ids[blank_mask] = audio_tokens_tensor
```

### 4. Supported Audio Formats

The implementation handles multiple audio input formats:

1. **Direct audio type**:
   ```json
   {"type": "audio", "audio": "<base64_data>"}
   ```

2. **Audio URL type**:
   ```json
   {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,<base64_data>"}}
   ```

3. **HTTP URL**:
   ```json
   {"type": "audio_url", "audio_url": {"url": "https://example.com/audio.wav"}}
   ```

4. **Data URL**:
   ```json
   {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,UklGR..."}}
   ```

## Expected Behavior

### Before Fix
```
Input: Audio file
input_ids: [BOS, ..., MEDIA_BEGIN, BLANK×188, MEDIA_END, ...]
Result:
  - Text: "" (empty)
  - EOS: Step 1 (logit=12.25)
```

### After Fix
```
Input: Audio file
input_ids: [BOS, ..., MEDIA_BEGIN, 152293, 152294, ..., 168158, MEDIA_END, ...]
Result:
  - Text: "Sure, I can count from 1 to 10..." (meaningful)
  - Audio: Generated (dual stream works)
  - EOS: After generating content
```

## Files Modified

1. `vllm_omni/entrypoints/openai/serving_chat.py`
   - Added `_is_kimi_audio_model()`
   - Added `_prepare_kimi_audio_inputs()`
   - Modified `_preprocess_chat()` to call Kimi Audio handler

2. `vllm_omni/model_executor/models/kimi_audio/kimi_audio_llm.py`
   - Already had code to check `_pending_audio_tokens`
   - Already had code to replace BLANK tokens with audio tokens

## Testing

Test with:
```bash
python test_kimi_audio_input.py
```

Expected:
- Text output should be meaningful (not empty)
- Audio output should be generated
- Model should not generate EOS immediately
