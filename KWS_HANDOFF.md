# KWS 交接契约（extract@sep → kws）

本文只定义机器可验证的跨仓契约。上游处理原理见 [TECHNICAL_FLOW.md](TECHNICAL_FLOW.md)，部署与验收命令见 [OPERATIONS.md](OPERATIONS.md)。

`kws` 不重跑本仓的 MossFormer；它读取完整 `VM_OUT`，复核各阶段结果并生成下游选路。`best_sep` 使用真值唤醒文本的 oracle CER，只是开发候选，不自动获得生产资格。

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
  best_sep/failed.jsonl
  best_sep/kws_handoff.json
  best_sep/{pos,neg}/{uid}.wav
  reports/audit_sep_run.json
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
- 中文 `streams.*.cer` 为无调拼音 CER，英文为规范化字符 CER；比赛严格字符 CER 在 `extract@main` 另算。
- 阶段内 `oracle_of`：min CER，平局 `original`；必须保留所有流的 hyp 与分项 CER 以便审计。
- 禁止用时长跳过、裁短或静默回退分离；输出与源时长差必须不超过 20 ms。

跨阶段导出规则为 `min oracle_cer → 平局优先非 original → 稳定阶段顺序`。这个规则是离线开发 oracle，不是运行时可直接获得的选择器。

`handoff` 只读取 `scripts/paths.py` 注册的 s1–s8 目录。`experiments/` 或任何未知阶段即使含有 `index.jsonl` 也不会进入 `best_sep`；新增正式阶段必须显式更新注册表、审计与 schema。

## `kws_handoff.json`

```json
{
  "schema": "kws_sep_handoff/v2",
  "extract_repo": "https://github.com/powerli2024/extract",
  "extract_branch": "sep",
  "mms_fa": false,
  "selector_within_stage": "oracle_cer_prefer_original",
  "selector_across_stages": "min_oracle_cer_prefer_sep_then_stage_order",
  "peak_norm": 0.7,
  "audio_length_policy": "full_utterance_no_truncation",
  "max_sep_sec": 0.0,
  "duration_audit_tolerance_sec": 0.02,
  "vm_out": "/root/autodl-tmp/kws_sep"
}
```

`kws` 的 `audit_sep_input.py --pos-neg $VM_OUT --check-duration --require-handoff`
会严格校验该文件和全部 s1–s8 结果。`mms_fa` 为 true、存在截断策略、覆盖不全或
输出时长不一致都会失败。

`best_sep/index.jsonl` **只含 ok=true**（wav 已拷到 `best_sep/{split}/{uid}.wav`）。缺失 wav 写到 `best_sep/failed.jsonl`，不进 kws 统计。

## 下游最低验收

```bash
python /root/kws/scripts/audit_sep_input.py \
  --pos-neg "$VM_OUT" \
  --expected-uids 1838 \
  --check-duration \
  --require-handoff \
  --out "$VM_OUT/reports/kws_input_audit.json"
```

只有审计退出码为 0、`n_fail=0`、当前 DatasetA 覆盖完整时，`kws` 才能开始独立比较。独立 KWS/声纹/Presence/比赛评测没有通过时，必须保持 `NO_GO`，不能因 `best_sep` 已成功导出而省略效果验收。
