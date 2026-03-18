# Kimi-Audio TTS Architecture Corrections

## Issue Identified ✅

The original implementation had **critical architectural mismatches** with the actual Kimi-Audio model structure shown in the diagram.

---

## 🔴 Original Issues

### 1. **Bifurcation Point WRONG**
- **Original**: Cloned after full backbone forward (all 28 layers)
- **Correct**: Clone AFTER layer 21, then:
  - Text path continues through layers 22-27
  - Audio path goes through MIMO layers 0-5

### 2. **Text Path MISSING**
- **Original**: No layers 22-27 processing
- **Correct**: Text path MUST continue through layers 22-27 → norm → lm_head

### 3. **mimo_output Head WRONG**
- **Original**: `ParallelLMHead(audio_output_vocab=16384, ...)`
- **Correct**: `ParallelLMHead(vocab_size=168448, ...)` 
  - Audio tokens are extracted by slicing logits[152064:168448]

### 4. **MIMO Layer Architecture**
- **Original**: Custom implementation
- **Correct**: Must EXACTLY match backbone layer architecture (GQA 28/4 heads)

---

## ✅ Corrected Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Input: Text Token IDs [B, L]                           │
│  → embed_tokens [168448, 3584]                          │
│  → inputs_embeds [B, L, 3584]                           │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Shared Backbone (Layers 0-21)                          │
│  - 22 layers with GQA (28 heads Q, 4 heads K/V)         │
│  - Hidden: [B, L, 3584]                                 │
└─────────────────────────────────────────────────────────┘
                          │
                          │ After Layer 21
                          ▼
                  ┌───────────────┐
                  │  CLONE        │
                  │  hidden_states│
                  └───────────────┘
                      │           │
                      │           │
        ┌─────────────┘           └─────────────┐
        ▼                                       ▼
┌───────────────────┐               ┌───────────────────┐
│  Text Path        │               │  Audio Path       │
│  Layers 22-27     │               │  MIMO Layers 0-5  │
│  (6 layers)       │               │  (6 layers)       │
│                   │               │                   │
│  → hidden_states  │               │  → mimo_hidden    │
│  → norm           │               │  → mimo_norm      │
│  → lm_head        │               │  → mimo_output    │
│  → text_logits    │               │  → audio_logits   │
│  [B, L, 168448]   │               │  [B, L, 168448]   │
│                   │               │                   │
│  Text tokens:     │               │  Audio tokens:    │
│  [0:152064]       │               │  [152064:168448]  │
└───────────────────┘               └───────────────────┘
```

---

## 🔧 Code Changes

### 1. Forward Method - Bifurcation Logic

**BEFORE (WRONG):**
```python
def forward(self, input_ids, ...):
    # Run through ALL 28 layers
    hidden_states = self.language_model.model(input_ids, ...)
    
    # Clone after everything
    mimo_hidden_states = hidden_states.clone()
    
    # Only MIMO path
    for mimo_layer in self.mimo_layers:
        mimo_hidden_states = mimo_layer(mimo_hidden_states)[0]
    
    return mimo_hidden_states
```

**AFTER (CORRECT):**
```python
def forward(self, input_ids, ...):
    # Layer-by-layer forward to enable bifurcation
    if inputs_embeds is None:
        hidden_states = self.language_model.model.embed_tokens(input_ids)
    else:
        hidden_states = inputs_embeds
    
    # Layers 0-21 (SHARED)
    for idx in range(22):
        layer = self.language_model.model.layers[idx]
        hidden_states = layer(hidden_states, position_ids=positions)[0]
    
    # BIFURCATION: Clone after layer 21
    mimo_hidden_states = hidden_states.clone()
    
    # TEXT PATH: Layers 22-27
    for idx in range(22, 28):
        layer = self.language_model.model.layers[idx]
        hidden_states = layer(hidden_states, position_ids=positions)[0]
    text_hidden_states = self.language_model.model.norm(hidden_states)
    
    # AUDIO PATH: MIMO layers 0-5
    for mimo_layer in self.mimo_layers:
        mimo_hidden_states = mimo_layer(mimo_hidden_states)[0]
    audio_hidden_states = self.mimo_norm(mimo_hidden_states)
    
    return {
        'text_hidden_states': text_hidden_states,
        'audio_hidden_states': audio_hidden_states,
    }
```

### 2. mimo_output Head - Full Vocab

**BEFORE (WRONG):**
```python
self.mimo_output = ParallelLMHead(
    self.config.audio_output_vocab,  # 16384 ❌
    self.config.hidden_size,
    ...
)
self.logits_processor = LogitsProcessor(self.config.audio_output_vocab)
```

**AFTER (CORRECT):**
```python
self.mimo_output = ParallelLMHead(
    self.config.vocab_size,  # 168448 ✅
    self.config.hidden_size,
    ...
)
self.logits_processor = LogitsProcessor(self.config.vocab_size)

# Audio tokens are extracted by:
# audio_logits = logits[:, :, 152064:168448]
```

### 3. MIMO Layer Architecture - Match Backbone

**BEFORE (Custom):**
```python
class KimiAudioMIMOAttention(nn.Module):
    # Generic attention
```

**AFTER (Exact Match):**
```python
class KimiAudioMIMOAttention(nn.Module):
    """EXACTLY matches backbone attention with GQA:
    - num_attention_heads: 28 (for Q)
    - num_key_value_heads: 4 (for K, V)
    - head_dim: 128 (3584 / 28)
    - num_key_value_groups: 7
    """
```

---

## 📊 Architecture Comparison

| Component | Original Model | My Implementation (Corrected) |
|-----------|---------------|-------------------------------|
| **Backbone Layers** | 28 (0-27) | ✅ 28 (0-27) |
| **Bifurcation Point** | After layer 21 | ✅ After layer 21 |
| **Text Path Layers** | 22-27 (6 layers) | ✅ 22-27 (6 layers) |
| **Audio Path Layers** | MIMO 0-5 (6 layers) | ✅ MIMO 0-5 (6 layers) |
| **Attention Type** | GQA (28/4) | ✅ GQA (28/4) |
| **Hidden Size** | 3584 | ✅ 3584 |
| **Intermediate Size** | 18944 | ✅ 18944 |
| **lm_head Output** | Full vocab [168448] | ✅ Full vocab [168448] |
| **mimo_output Output** | Full vocab [168448] | ✅ Full vocab [168448] |
| **Text Token Range** | [0:152064] | ✅ [0:152064] |
| **Audio Token Range** | [152064:168448] | ✅ [152064:168448] |

---

## 🧪 Testing Checklist

After corrections, verify:

- [ ] **Layer 21 Bifurcation**: Confirm hidden_states cloned after layer 21
- [ ] **Text Path Execution**: Layers 22-27 are processed
- [ ] **MIMO Layer Count**: Exactly 6 MIMO layers
- [ ] **GQA Configuration**: 28 heads for Q, 4 heads for K/V
- [ ] **mimo_output Vocab**: Full 168448, not 16384
- [ ] **Weight Loading**: All weights load correctly (including layers 22-27)
- [ ] **Audio Token Generation**: Tokens in range [152064:168448]
- [ ] **End-to-End TTS**: Generate audible audio

---

## 📝 Files Updated

| File | Change |
|------|--------|
| `kimi_audio_talker.py` | Complete rewrite with correct bifurcation logic |
| `configuration_kimi_audio_tts.py` | Updated docstrings for full vocab output |
| `kimi_audio_talker.py.bak` | Backup of original (incorrect) implementation |

---

## 🎯 Key Takeaways

1. **Bifurcation happens AFTER layer 21**, not after full backbone
2. **Text path continues** through layers 22-27 (important for unified ASR+TTS)
3. **mimo_output uses FULL vocab** (168448), audio tokens are sliced
4. **MIMO layers must EXACTLY match** backbone architecture (GQA 28/4)

---

**Correction Date:** 2026-03-12  
**Status:** Corrected and ready for testing ✅
