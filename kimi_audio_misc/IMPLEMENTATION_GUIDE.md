# Kimi Audio 音频分词修复 - 实施指南

## 日期：2026-06-30
## 状态：🔴 部分实现 - 需要进一步工作

---

## 问题总结

### 根本原因
模型立即生成 EOS（空文本输出）是因为：
- **模型期望**：来自 Glm4Tokenizer 的离散音频 token（例如 [152293, 152294, ...]）
- **vllm-omni 提供**：BLANK token（151666）
- **结果**：模型无法识别 BLANK 模式 → 以高置信度生成 EOS（logit=12.25）

### 已完成的修复
✅ 添加了 Glm4Tokenizer 初始化（第 287-310 行）
✅ 添加了 `_tokenize_audio_to_discrete_tokens` 方法（第 539-598 行）
✅ 修改了 `_parse_and_validate_audio_input` 以分词音频（第 520-544 行）
✅ 修改了 `forward()` 以用实际音频 token 替换 BLANK token（第 306-340 行）

### 未解决的问题
❌ **原始音频数据未传递到模型**

当前流程：
```
API 接收 audio_url (base64)
  ↓
API 服务器解码音频
  ↓
HF 处理器提取 whisper 特征
  ↓
模型接收 whisper_features（但没有原始音频！）
```

问题：`_parse_and_validate_audio_input` 检查 `kwargs.get("audio", None)`，但 HF 处理器只传递 whisper 特征，不传递原始音频。

---

## 需要完成的工作

### 方案 1：修改多模态处理器以传递原始音频（推荐）

**文件**：`vllm_omni/entrypoints/openai/serving_chat.py` 或多模态处理器

**步骤**：
1. 找到 HF 处理器被调用的位置
2. 修改以同时传递原始音频和 whisper 特征
3. 确保原始音频在 `kwargs` 中作为 `"audio"` 键可用

**示例代码**：
```python
# 在 HF 处理器调用处
processed = hf_processor(audio=raw_audio, ...)
# 确保 raw_audio 被传递到模型的 _parse_and_validate_audio_input
```

### 方案 2：在多模态处理器中分词音频

**文件**：上游 vLLM 的 `KimiAudioMultiModalProcessor`

**步骤**：
1. 创建自定义多模态处理器继承自 `KimiAudioMultiModalProcessor`
2. 在处理器中添加音频分词逻辑
3. 将离散 token 作为额外的多模态数据传递

**示例代码**：
```python
class CustomKimiAudioMultiModalProcessor(KimiAudioMultiModalProcessor):
    def _call_hf_processor(self, prompt, mm_data, ...):
        # 提取 whisper 特征
        hf_inputs = super()._call_hf_processor(prompt, mm_data, ...)
        
        # 分词音频
        if "audio" in mm_data:
            audio_tokens = self.audio_tokenizer.tokenize(mm_data["audio"])
            hf_inputs["audio_tokens"] = audio_tokens
        
        return hf_inputs
```

### 方案 3：在 API 服务器层面分词

**文件**：`vllm_omni/entrypoints/openai/serving_chat.py`

**步骤**：
1. 在 API 服务器中加载 Glm4Tokenizer
2. 接收音频时立即分词
3. 将离散 token 作为请求的一部分传递

**示例代码**：
```python
# 在 create_chat_completion 中
if audio_url:
    audio_data = decode_audio(audio_url)
    audio_tokens = self.audio_tokenizer.tokenize(audio_data)
    # 将 audio_tokens 添加到请求中
```

---

## 验证步骤

完成修复后，验证：

1. **检查日志**：
```bash
pm2 logs kimi-audio-8091 | grep -E "(Tokenizing audio|Tokenized audio|Stored.*discrete|Replaced BLANK)"
```

期望输出：
```
[KimiAudio] Tokenizing audio: /tmp/tmpXXX.wav
[KimiAudio] ✅ Tokenized audio into 188 discrete tokens
[KimiAudio] First 5 tokens: [152293, 152294, 152301, ...]
[KimiAudio] ✅ Stored 188 discrete audio tokens
[KimiAudio] forward: Found 188 BLANK tokens in input_ids
[KimiAudio] ✅ Replaced BLANK tokens with discrete audio tokens
```

2. **运行测试**：
```bash
python test_kimi_audio_input.py
```

期望结果：
- 文本输出：有意义的响应（非空）
- 音频输出：生成（双流工作）
- finish_reason: "stop"（但生成内容后，不是立即）

3. **检查响应**：
```json
{
  "choices": [
    {
      "message": {
        "content": "Sure, I can count from 1 to 10...",  // 非空！
        "audio": { ... }  // 也生成
      },
      "finish_reason": "stop"
    }
  ]
}
```

---

## 关键技术细节

### Glm4Tokenizer 用法
```python
from transformers import AutoTokenizer

audio_tokenizer = AutoTokenizer.from_pretrained(
    "THUDM/glm-4-voice-tokenizer",
    trust_remote_code=True
)

# 分词音频
wav_tokens = audio_tokenizer.tokenize(audio_path="/path/to/audio.wav")

# 添加偏移量得到实际 token ID
wav_tokens = wav_tokens + 152064  # token_offset

# 转换为列表
wav_tokens_list = wav_tokens.squeeze(0).cpu().numpy().tolist()
```

### BLANK Token 替换逻辑
```python
# 在 forward() 中
if self._current_audio_tokens is not None:
    blank_mask = (input_ids == self._blank_token_id)  # 151666
    num_blank_positions = blank_mask.sum().item()
    
    if len(self._current_audio_tokens) == num_blank_positions:
        audio_tokens_tensor = torch.tensor(
            self._current_audio_tokens,
            dtype=input_ids.dtype,
            device=input_ids.device
        )
        input_ids = input_ids.clone()
        input_ids[blank_mask] = audio_tokens_tensor
```

### 双流融合
```python
# 在 embed_input_ids() 中
# 1. 嵌入文本 token（包括音频 token）
text_emb = self.model.model.embed_tokens(input_ids)

# 2. 处理 whisper 连续特征
audio_features = self._process_audio_input(whisper_features)

# 3. 在音频位置融合
combined_emb = text_emb[multimodal_positions] + audio_features
combined_emb = combined_emb * (2 ** 0.5)  # √2 缩放
text_emb[multimodal_positions] = combined_emb
```

---

## 文件修改清单

### 已修改
1. `vllm_omni/model_executor/models/kimi_audio/kimi_audio_llm.py`
   - 添加 Glm4Tokenizer 初始化（~25 行）
   - 添加 `_tokenize_audio_to_discrete_tokens` 方法（~60 行）
   - 修改 `_parse_and_validate_audio_input`（~15 行）
   - 修改 `forward()` 以替换 BLANK token（~35 行）

### 需要修改
2. `vllm_omni/entrypoints/openai/serving_chat.py` 或多模态处理器
   - 传递原始音频数据到模型
   - 或在此层面执行音频分词

---

## 测试场景

### 测试 1：单请求
```bash
python test_kimi_audio_input.py
```
期望：文本 + 音频输出

### 测试 2：批量请求
```bash
# 发送 2 个并发请求
curl http://localhost:8091/v1/chat/completions ... &
curl http://localhost:8091/v1/chat/completions ... &
wait
```
期望：每个请求独立生成音频输出

### 测试 3：顺序请求
```bash
# 发送请求 1，等待完成，发送请求 2
curl http://localhost:8091/v1/chat/completions ...
curl http://localhost:8091/v1/chat/completions ...
```
期望：第二个请求不继承第一个请求的状态

---

## 已知问题

### 问题 1：原始音频未传递
- **症状**：`_tokenize_audio_to_discrete_tokens` 从未被调用
- **原因**：HF 处理器不传递原始音频到 `kwargs["audio"]`
- **解决**：修改多模态处理管道以传递原始音频

### 问题 2：Token 计数不匹配
- **潜在问题**：BLANK token 数量可能与音频 token 数量不匹配
- **检查**：在 `forward()` 中记录 `num_blank_positions` 和 `len(self._current_audio_tokens)`
- **解决**：如果不匹配，需要调整分词逻辑或提示构造

---

## 下一步行动

### 优先级 1：传递原始音频（关键）
选择上述方案之一（推荐方案 1 或 2）来传递原始音频数据到模型。

### 优先级 2：测试和验证
- 运行 `test_kimi_audio_input.py`
- 检查日志以验证分词和替换是否发生
- 验证文本输出是否有意义

### 优先级 3：批量推理测试
- 测试多个并发请求
- 验证每个请求都有独立的音频状态
- 确保没有状态泄漏

---

## 联系和资源

### 调试日志
```bash
pm2 logs kimi-audio-8091
```

### 测试脚本
- `test_kimi_audio_input.py` - 单请求测试
- `test_kimi_audio_batched.py` - 批量测试（需要创建）

### 参考实现
- `/root/learning/Kimi-Audio/kimia_infer/api/prompt_manager.py` - 参考分词逻辑
- `/root/learning/Kimi-Audio/kimia_infer/utils/sampler.py` - KimiASampler 实现

### 文档
- `kimi_audio_misc/FINAL_STATUS.md` - 最终状态报告
- `kimi_audio_misc/audio_tokenization_fix_summary.md` - 修复总结
- `kimi_audio_misc/root_cause_and_solution.md` - 根本原因分析

---

## 结论

已经实现了音频分词和 BLANK token 替换的核心逻辑，但由于原始音频数据未传递到模型，修复尚未完成。

**关键缺失组件**：需要将原始音频数据从 API 服务器传递到模型的多模态处理方法。

**估计工作量**：2-4 小时（取决于选择的方案）

**风险**：中等 - 需要理解 vLLM 的多模态处理框架

**下一步**：选择方案 1、2 或 3 来传递原始音频数据，然后测试验证。
