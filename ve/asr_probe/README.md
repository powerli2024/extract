# asr_probe：全量 pos/neg × 流 ASR（main 分支历史参考）

> 本工具属于 VE/CMD 诊断，不在 `extract@sep` 执行。sep 分支的 KWS ASR/CER 链见 [TECHNICAL_FLOW.md](../../TECHNICAL_FLOW.md)。

**不做拒识、不跑 TSE。** 对每条 CMD 的 mix / 一次分离 / 级联分离轨跑 Qwen3-ASR，写出 jsonl + 分析 JSON。

| 臂 | 波形 |
|----|------|
| `no_sep` | 原始 CMD（或 `d1/.../mix.wav`） |
| `sep_once` | `sep_streams/d1/{split}/{uid}/d1_*.wav` |
| `sep_multi` | `sep_streams/d2/{split}/{uid}/` 下非 mix/peak 轨 |

- **pos**：相对 `cmd_text` 算 CER（与 VE `asr_cer` 同一归一化）
- **neg**：无识别文本，只存转写；分析空转写率、长度、是否像唤醒词

## 跑

```bash
cd /root/extract/ve/asr_probe
source ../.env_ve
./run_asr_probe.sh
# 冒烟
LIMIT=8 ./run_asr_probe.sh
```

默认输出：`/root/autodl-tmp/asr_probe/`

```text
$ASR_PROBE_OUT/
  asr_results.jsonl    # 一行=一条 (uid, arm, stream)
  analysis.json
  analysis.md
```

断点：默认 `--resume`。改模型后换 `OUT` 或删 jsonl。

`LIMIT=N` 表示 **pos / neg 各 N 条 utt**。`sep_once` / `sep_multi` 默认不重复转 mix（与 `no_sep` 去重）；若要对齐 Presence 打分臂，加 `--include-mix-in-sep`。

分析看 `analysis.md` / `analysis.json`：pos 的 mix CER vs 分离 oracle（每条 utt 取 min CER，乐观上界）vs 全部轨 pool；neg 的空转写、长度、mix 空但分离轨出字。

下一刀离线验收（锁定门控 + CER=1 桶 + camp 否决）：

```bash
python asr_probe/scripts/eval_next_lift.py
# 或仓库根: ./run_next_lift.sh t4
```

产物默认 `datasetA/sssss/next_lift_eval.json`。T1 另一套转写用 `--alt-asr asr_results.jsonl`。
