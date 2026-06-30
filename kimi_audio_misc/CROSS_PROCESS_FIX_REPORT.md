# Kimi Audio 双路流修复报告

## 问题描述

Kimi Audio 模型在推理时无法正确生成文本和音频的双路流输出。具体表现为：
- 文本输出为空或只有 EOS token
- 音频生成正常
- 日志显示 "No pending audio tokens"，表明模型没有接收到音频 token

## 根本原因

系统采用跨进程架构：
- **API Server 进程** (serving_chat.py)：负责接收请求、处理音频、tokenize
- **Model 进程** (StageEngineCoreProc)：负责实际推理

原始实现使用类变量 `KimiAudioLLMForConditionalGeneration._pending_audio_tokens` 在 API Server 中存储音频 token，但由于两个进程是独立的，类变量无法跨进程共享，导致 Model 进程无法访问这些 token。

## 解决方案

### 1. 修改音频数据传递方式

**文件**: `vllm_omni/entrypoints/openai/serving_chat.py`

**改动**:
```python
# 原来：返回空字典，音频通过类变量传递（失败）
return stripped_messages, {}

# 修改后：返回原始音频数据，通过 deferred_multi_modal_data 传递
return stripped_messages, {"audio": materialized_audio}
```

这样音频数据会通过 `additional_information["deferred_multi_modal_data"]["audio"]` 传递到 Model 进程。

### 2. 修复音频 tokenize 方法

**文件**: `vllm_omni/model_executor/models/kimi_audio/kimi_audio_llm.py`

**改动**:
```python
# 原来：将 bytes 当作文件路径处理（错误）
if isinstance(raw_audio, (str, bytes)):
    audio_path = raw_audio

# 修改后：正确处理 bytes 类型
if isinstance(raw_audio, bytes):
    # 将音频 bytes 保存到临时文件
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        f.write(raw_audio)
        temp_path = f.name
    audio_path = temp_path
elif isinstance(raw_audio, str):
    audio_path = raw_audio
```

同时更新了临时文件清理逻辑：
```python
# 清理临时文件（包括从 bytes 创建的文件）
if isinstance(raw_audio, (bytes, np.ndarray, torch.Tensor)) and 'temp_path' in locals():
    try:
        os.unlink(temp_path)
    except Exception:
        pass
```

## 工作流程

1. **API Server 进程**:
   - 接收包含音频的请求
   - 解码 base64 音频数据为 bytes
   - 使用 Glm4Tokenizer tokenize 音频（生成 audio_tokens）
   - 将原始音频 bytes 通过 `deferred_multi_modal_data` 返回

2. **Engine Core 进程**:
   - 接收请求和 `additional_information`
   - 从 `deferred_multi_modal_data["audio"]` 获取原始音频 bytes
   - 调用 `_tokenize_audio_to_discrete_tokens()` 进行 tokenize
   - 将 audio bytes 保存到临时文件
   - 使用 Glm4Tokenizer 读取并 tokenize
   - 将 token 存储到 `self._current_audio_tokens`

3. **Model 推理**:
   - 在 `embed_input_ids()` 中融合文本和音频 embedding
   - 生成文本和音频的双路流输出

## 测试结果

使用测试音频 `/root/learning/Kimi-Audio/test_audios/qa_example.wav` 进行测试：

```bash
python test_kimi_audio_input.py
```

**结果**:
- ✅ **文本输出**: "A person is playing a drum set with a bass drum and a snare drum."
- ✅ **音频输出**: 成功生成（base64 WAV 数据）
- ✅ **双路流**: 文本和音频同时正常输出

## 日志验证

关键日志信息：
```
[ServingChat] _prepare_kimi_audio_inputs called
[ServingChat] Found 1 audio parts
[ServingChat] Tokenized audio into 47 tokens
[ServingChat] ✅ Stored 47 audio tokens for model
[KimiAudio] forward: Found deferred_multi_modal_data with keys: ['audio']
[KimiAudio] forward: Found 1 audio items in deferred data
[KimiAudio] ✅ forward: Tokenized audio into 47 discrete tokens
[KimiAudio] embed_input_ids: Added 9 audio token embeddings
```

## 代码质量改进

在修复过程中，还进行了以下代码质量改进：

1. **移除冗余导入**: 将 `os`, `tempfile`, `requests`, `traceback` 等移到文件顶部
2. **修复裸 except**: 将所有 `except:` 改为 `except Exception:`
3. **改进临时文件管理**: 确保所有临时文件都被正确清理

## 提交信息

```
commit f90d5689
[Bugfix] Fix cross-process audio token passing for Kimi Audio

- Pass raw audio bytes through deferred_multi_modal_data instead of class variable
- Fix _tokenize_audio_to_discrete_tokens to handle bytes properly
- Enable proper dual stream generation (text + audio) across processes
```

## 服务器状态

- **PM2 服务**: kimi-audio-8091
- **状态**: 在线运行
- **端口**: 8091
- **GPU**: GPU 6 (CUDA_VISIBLE_DEVICES=6)

## 结论

通过修改跨进程数据传递方式，成功解决了 Kimi Audio 的双路流生成问题。现在模型可以正确地同时生成文本和音频输出，符合预期行为。
