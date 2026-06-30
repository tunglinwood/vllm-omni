# Kimi Audio 双流架构不匹配问题

## 日期：2026-06-30
## 状态：🔴 关键问题发现 - 架构不匹配

---

## 根本问题

### 参考实现（Kimi-Audio 官方）
```python
# 双流架构
audio_input_ids = [BOS, USER_START, MEDIA_BEGIN, 152293, 152294, ..., 168158, MEDIA_END, ...]
                                         ↑
                                    实际音频 token（来自 Glm4Tokenizer）

text_input_ids = [BOS, USER_START, MEDIA_BEGIN, BLANK, BLANK, ..., BLANK, MEDIA_END, ...]
                                         ↑
                                    BLANK token 占位符

# 模型接收两个独立的流
audio_logits, text_logits, past_key_values = self.alm.forward(
    input_ids=audio_input_ids,      # 音频流：实际音频 token
    text_input_ids=text_input_ids,  # 文本流：文本 token + BLANK 占位
    whisper_input_feature=whisper_features,  # 连续特征
    ...
)
```

### vllm-omni 实现
```python
# 单流架构
input_ids = [BOS, USER_START, MEDIA_BEGIN, BLANK, BLANK, ..., BLANK, MEDIA_END, ...]
                                         ↑
                                    BLANK token（151666）

# 模型只接收一个流
outputs = model.forward(
    input_ids=input_ids,  # 只有 BLANK token
    multimodal_embeddings=whisper_features,  # 连续特征通过融合添加
    ...
)
```

---

## 问题分析

### 模型期望
- 训练时：音频流包含**实际音频 token**（152293, 152294, ...）
- 推理时：也应该看到**实际音频 token**

### vllm-omni 提供
- 音频位置：**BLANK token**（151666）
- 连续特征：通过融合添加到嵌入中

### 结果
- 模型看到 BLANK token → 不识别模式 → 立即生成 EOS（logit=12.25）
- 文本输出：空（""）

---

## 解决方案

### 方案 1：在 forward() 中替换 BLANK token（当前尝试）

**思路**：
1. 在 forward() 中从 `additional_information["deferred_multi_modal_data"]["audio"]` 提取原始音频
2. 使用 Glm4Tokenizer 分词得到实际音频 token
3. 用实际音频 token 替换 input_ids 中的 BLANK token

**代码**：
```python
def forward(self, input_ids, ..., additional_information: Optional[dict] = None):
    # 提取音频并分词
    if additional_information is not None and self._current_audio_tokens is None:
        deferred_data = additional_information.get("deferred_multi_modal_data", {})
        audio_data_list = deferred_data.get("audio", [])
        if audio_data_list:
            raw_audio = audio_data_list[0]
            discrete_tokens = self._tokenize_audio_to_discrete_tokens(raw_audio)
            self._current_audio_tokens = discrete_tokens
    
    # 替换 BLANK token
    if self._current_audio_tokens is not None:
        blank_mask = (input_ids == self._blank_token_id)
        if len(self._current_audio_tokens) == blank_mask.sum().item():
            input_ids = input_ids.clone()
            input_ids[blank_mask] = torch.tensor(self._current_audio_tokens, ...)
    
    # 继续正常前向传播
    ...
```

**问题**：
- `additional_information` 似乎没有被传递到 forward()
- 需要确认 vllm-omni 的多模态处理管道是否支持这个参数

**状态**：⚠️ 实现完成但未验证

---

### 方案 2：在 API 服务器层面替换（推荐）

**思路**：
1. 在 `serving_chat.py` 中接收音频时立即分词
2. 将实际音频 token 传递给多模态处理器
3. 修改 prompt 构造逻辑以使用实际音频 token 而不是 BLANK token

**文件**：`vllm_omni/entrypoints/openai/serving_chat.py`

**代码**：
```python
async def _prepare_multistage_multimodal_inputs(self, ...):
    # 解码音频
    audio_data = await self._materialize_deferred_multimodal_parts(...)
    
    # 分词音频
    audio_tokens = self.audio_tokenizer.tokenize(audio_data)
    audio_tokens = audio_tokens + self._token_offset  # 152064
    
    # 将 audio_tokens 传递给多模态处理器
    multi_modal_data["audio_tokens"] = audio_tokens
    
    return stripped_messages, multi_modal_data
```

**修改 chat template**：
```jinja
{# 不是使用 BLANK token #}
{{ media_begin }}{{ blank }}{{ media_end }}

{# 而是使用实际音频 token #}
{{ media_begin }}{% for token in audio_tokens %}{{ token }}{% endfor %}{{ media_end }}
```

**优点**：
- 在最早期阶段处理音频
- 避免在 forward() 中做复杂逻辑
- 更符合 vllm-omni 的架构

**缺点**：
- 需要修改 chat template 以支持动态 token
- 可能需要修改多模态处理器的接口

**状态**：💡 设计完成，未实现

---

### 方案 3：修改多模态处理器

**思路**：
1. 创建自定义多模态处理器继承自 `KimiAudioMultiModalProcessor`
2. 在处理器中调用 Glm4Tokenizer 分词音频
3. 将实际音频 token 作为多模态数据的一部分传递

**文件**：新建 `vllm_omni/model_executor/models/kimi_audio/multimodal_processor.py`

**代码**：
```python
class CustomKimiAudioMultiModalProcessor(KimiAudioMultiModalProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 加载 Glm4Tokenizer
        from transformers import AutoTokenizer
        self.audio_tokenizer = AutoTokenizer.from_pretrained(
            "THUDM/glm-4-voice-tokenizer",
            trust_remote_code=True
        )
    
    def _call_hf_processor(self, prompt, mm_data, ...):
        # 调用父类处理器获取 whisper 特征
        hf_inputs = super()._call_hf_processor(prompt, mm_data, ...)
        
        # 分词音频
        if "audio" in mm_data:
            audio_tokens = self.audio_tokenizer.tokenize(mm_data["audio"])
            audio_tokens = audio_tokens + 152064  # token_offset
            hf_inputs["audio_tokens"] = audio_tokens
        
        return hf_inputs
```

**优点**：
- 符合 vllm-omni 的多模态处理架构
- 可以在处理器中集中管理音频分词
- 不需要修改 API 服务器代码

**缺点**：
- 需要理解 vllm-omni 的多模态处理框架
- 可能需要修改模型实现以接收 audio_tokens

**状态**：💡 设计完成，未实现

---

## 当前状态

### 已实现
1. ✅ Glm4Tokenizer 加载和初始化
2. ✅ `_tokenize_audio_to_discrete_tokens` 方法
3. ✅ BLANK token 替换逻辑（在 forward() 中）
4. ✅ 融合公式修正（ADD + √2 缩放）

### 未解决
1. ❌ `additional_information` 未传递到 forward()
2. ❌ 音频分词代码未执行
3. ❌ BLANK token 未被替换
4. ❌ 模型仍看到 BLANK token → 生成 EOS

---

## 下一步行动

### 优先级 1：确认 additional_information 是否被传递

**检查点**：
1. 在 forward() 开头添加日志：`print(f"additional_information: {additional_information}")`
2. 运行测试并检查日志
3. 如果 `additional_information is None`，说明 vllm-omni 不支持这个参数

**如果支持**：
- 检查 deferred_multi_modal_data 是否包含音频数据
- 检查音频分词是否执行
- 检查 BLANK token 是否被替换

**如果不支持**：
- 转向方案 2 或 3
- 需要修改 vllm-omni 的多模态处理管道

### 优先级 2：实现替代方案

如果方案 1 不可行，选择方案 2 或 3：
- 方案 2：在 API 服务器层面处理（更直接）
- 方案 3：在多模态处理器中处理（更符合架构）

---

## 技术细节

### Glm4Tokenizer 用法
```python
from transformers import AutoTokenizer

audio_tokenizer = AutoTokenizer.from_pretrained(
    "THUDM/glm-4-voice-tokenizer",
    trust_remote_code=True
)

# 分词音频文件
wav_tokens = audio_tokenizer.tokenize(audio_path="/path/to/audio.wav")

# 添加偏移量得到实际 token ID
wav_tokens = wav_tokens + 152064  # kimia_token_offset

# 转换为列表
wav_tokens_list = wav_tokens.squeeze(0).cpu().numpy().tolist()
# 结果：[152293, 152294, 152301, ...]
```

### Token 范围
- BLANK token: 151666
- 音频 token: 152064 - 168447（16384 个 token）
- EOS token: 151667

### 融合公式
```python
# 正确公式（已实现）
combined_emb = (discrete_emb + continuous_emb) * sqrt(2)

# 错误公式（之前使用）
combined_emb = continuous_emb  # 替换而不是融合
```

---

## 预期结果

### 修复前
```
输入：音频文件
input_ids: [BOS, ..., MEDIA_BEGIN, BLANK×188, MEDIA_END, ...]
结果：
  - 文本：""（空）
  - EOS：第 1 步立即生成（logit=12.25）
```

### 修复后
```
输入：音频文件
input_ids: [BOS, ..., MEDIA_BEGIN, 152293, 152294, ..., 168158, MEDIA_END, ...]
结果：
  - 文本："Sure, I can count from 1 to 10..."（有意义）
  - 音频：生成（双流工作）
  - EOS：生成内容后
```

---

## 参考资源

### 参考实现
- `/root/learning/Kimi-Audio/kimia_infer/api/prompt_manager.py` - 双流构造
- `/root/learning/Kimi-Audio/kimia_infer/api/kimia.py` - 双流推理

### vllm-omni 代码
- `vllm_omni/entrypoints/openai/serving_chat.py` - API 服务器
- `vllm_omni/model_executor/models/kimi_audio/kimi_audio_llm.py` - 模型实现
- `vllm_omni/model_executor/stage_input_processors/kimi_audio.py` - 多模态处理器

### 文档
- `FINAL_FIX_IMPLEMENTATION.md` - 当前实现总结
- `kimi_audio_model_arch.md` - 模型架构说明
