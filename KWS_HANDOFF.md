# KWS 交接契约（extract@sep → kws）

`kws` 不跑 MossFormer。它只读本分支 `VM_OUT`。

## 目录

```text
$VM_OUT/
  kws_handoff.json
  pos/s1_onnx_full/index.jsonl
  pos/s1_onnx_full/wav/{uid}_peak.wav
  pos/s1_onnx_full/wav/{uid}_spk1.wav
  pos/s1_onnx_full/wav/{uid}_spk2.wav
  pos/s2_cv_full/...
  pos/s3_onnx_cascade/wav/{uid}_spk1_r1.wav  # cascade
  neg/...
  best_sep/index.jsonl
  best_sep/{pos,neg}/{uid}.wav
```

`oracle_stream=original` 对应 wav tag `peak`。

## 阶段 index 一行（最低字段）

```json
{
  "uid": "pos_0",
  "split": "pos",
  "wake_text": "你好科慕",
  "oracle_stream": "spk2",
  "oracle_cer": 0.0,
  "streams": {
    "original": {"hyp": "...", "cer": 1.11, "cer_char": 1.0, "cer_py": 1.11},
    "spk1": {"hyp": "...", "cer": 1.0, "cer_char": 1.0, "cer_py": 1.0},
    "spk2": {"hyp": "...", "cer": 0.0, "cer_char": 0.0, "cer_py": 0.0}
  }
}
```

- **无 MMS-FA。** `streams.*.cer` 来自 Qwen3 + `scripts/cer_pinyin.py`。
- 阶段内 `oracle_of`：min CER，平局 `original`。
- 禁止用时长跳过分离。

## `kws_handoff.json`

```json
{
  "schema": "kws_sep_handoff/v1",
  "extract_repo": "https://github.com/powerli2024/extract",
  "extract_branch": "sep",
  "mms_fa": false,
  "selector_within_stage": "oracle_cer_prefer_original",
  "selector_across_stages": "min_oracle_cer_prefer_sep_then_stage_order",
  "peak_norm": 0.7,
  "vm_out": "/root/autodl-tmp/kws_sep"
}
```

`kws` 的 `rebuild_best_sep.py --pos-neg $VM_OUT` 会校验该文件。`mms_fa` 为 true 则失败。

`best_sep/index.jsonl` **只含 ok=true**（wav 已拷到 `best_sep/{split}/{uid}.wav`）。缺失 wav 写到 `best_sep/failed.jsonl`，不进 kws 统计。
