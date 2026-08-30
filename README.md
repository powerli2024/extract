# extract `sep`：KWS 分离、CER 选轨与下游交接

本仓库是 [`powerli2024/extract`](https://github.com/powerli2024/extract) 的 `sep` 分支，唯一职责是：对 DatasetA 的 KWS 注册音频生成多种完整时长分离候选，用 Qwen3-ASR 计算开发期 oracle CER，导出可审计的 `best_sep`，再交给 `kws` 做独立选路与验证。

> 本分支不运行 Presence、CMD 提取或比赛 ASR。`./ve.sh` 会主动退出；这些步骤必须在旁边的 `/root/extract`（`main` 分支）运行。`ve/` 下的 Markdown 仅保留为 `main` 分支历史参考，不是本分支操作手册。

## 文档导航

| 文档 | 解决的问题 |
|---|---|
| [TECHNICAL_FLOW.md](TECHNICAL_FLOW.md) | s1–s8 到底做什么，CER、门控、选轨和续跑怎样工作 |
| [OPERATIONS.md](OPERATIONS.md) | AutoDL 安装、冒烟、全量重跑、严格验收和故障处理 |
| [KWS_HANDOFF.md](KWS_HANDOFF.md) | `extract-sep → kws` 的目录、JSONL 与审计契约 |
| [IMPROVEMENTS.md](IMPROVEMENTS.md) | 当前风险、分阶段改进方案和 Go/No-Go 标准 |
| [DAE_TSE.md](DAE_TSE.md) | DAE-TSE 的统一环境接入、运行命令和三级验收门 |
| [environment/README_CUDA124.md](environment/README_CUDA124.md) | Ubuntu 20.04 + CUDA 12.4 单环境构建、版本清单与验收 |

## 一张图看懂边界

```text
DatasetA/{pos,neg}.jsonl + {pos,neg}/kws_*.wav
                         │
                         ▼
                 collect：固定 uid / 文本 / 路径
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
 s1 ONNX 一阶全量                    s2 ClearVoice 一阶全量
        │                                 │
        ├──► s3 ONNX cascade              ├──► s4 ClearVoice cascade
        │                                 │
        ├──► s5 CV 重做原波                ├──► s7 ONNX refine
        └──► s6 ONNX refine               └──► s8 CV refine
               仅对父阶段高 CER 子集运行，a/b/c 同 cohort 时去重
                         │
                         ▼
            compare / eval / strict audit
                         │
                         ▼
     handoff：跨阶段最小 CER → best_sep + kws_handoff.json
                         │
                         ▼
       /root/kws：独立重建、声纹/下游指标复核、Go/No-Go
                         │
                         ▼
       /root/extract@main：Presence / CMD / 真实比赛评测
```

可选 DAE-TSE 默认禁用；仅显式运行 `run_dae_tse.sh` 才从 raw 或某个已冻结阶段 winner 分叉，输出到 `$VM_OUT/experiments/`，不会进入默认主链或 handoff。所有分离模型调用前统一执行 mono/16 kHz、`[-1,1]` 限幅和峰值 `0.70` 预处理。

这里的 `best_sep` 是开发期候选，不等于生产批准结果。阶段内和跨阶段都使用了已知唤醒文本的 oracle CER；正式采用仍需在冻结的 KWS、Presence/CMD 和比赛指标上独立复核。

## 阶段速查

| 阶段 | 输入 | 分离器 | 候选 | 用途 |
|---|---|---|---|---|
| s1 | 原始 KWS | MossFormer2 ONNX | original、spk1、spk2 | 低成本全量基线 |
| s2 | 原始 KWS | ClearVoice MossFormer2 | original、spk1、spk2 | 第二个一阶后端 |
| s3 | s1 的 spk1/spk2 | ONNX | original、父两轨、二阶四轨 | ONNX cascade 对照 |
| s4 | s2 的 spk1/spk2 | ClearVoice | 同上 | ClearVoice cascade 对照 |
| s5 | s1 高 CER 子集的 original | ClearVoice | original、spk1、spk2 | 换后端抢救 |
| s6 | s1 高 CER 子集的两轨 | ONNX | original、父两轨、二阶四轨 | ONNX refine |
| s7 | s2 高 CER 子集的两轨 | ONNX | 同上 | CV→ONNX refine |
| s8 | s2 高 CER 子集的两轨 | ClearVoice | 同上 | CV→CV refine |

中文使用无调拼音 CER，英文使用规范化字符 CER。阶段内取最小 CER，平局保留 `original`；跨阶段先取最小 CER，平局优先分离轨，再按稳定的阶段顺序决定。详细规则见 [TECHNICAL_FLOW.md](TECHNICAL_FLOW.md)。

## AutoDL 最短可运行链

```bash
cd /root
git clone -b sep https://github.com/powerli2024/extract.git extract-sep
cd /root/extract-sep

export DATA_DIR=/root/autodl-tmp/datasetA
export VM_OUT=/root/autodl-tmp/kws_sep_v2
export MOSS_CKPT_DIR=/root/autodl-tmp/checkpoints
export ASR_MODEL_DIR=/root/autodl-tmp/Qwen3-ASR-1.7B

bash ./setup_env.sh
source ./env.sh
./download_models.sh
bash ./check_env.sh
bash ./rerun_and_audit.sh --full
```

推荐每轮使用新的 `VM_OUT`。同一目录只续跑签名和配置一致的任务；不要把旧目录的 index、ASR 或 WAV 手工拷入新实验。

## 完成定义

一次全量运行只有同时满足下列条件才算完成：

- DatasetA 共 1,838 个 UID（pos 1,364、neg 474）全部进入审计；若数据版本变化，应以当前 manifest 为准并同步调整期望数。
- s1–s4 全量 index、WAV、CER 和音频时长完整；s5–s8 的门控子集与父阶段阈值计划一致。
- 中文记录为 `metric=pinyin`，英文为 `metric=char`；错误、缺 WAV、跨 split UID 和超过 20 ms 的时长差均为 0。
- `best_sep/index.jsonl`、`best_sep/{pos,neg}/*.wav` 与 `kws_handoff.json` 一致，`n_fail=0`。
- `/root/kws/scripts/audit_sep_input.py --check-duration --require-handoff` 通过；后续独立验证未通过时，结论仍应写 `NO_GO`。

`--limit` 只验证流程可运行，不能用于选最优阶段或宣称指标提升。

## 关键产物

```text
$VM_OUT/
  pos/、neg/                         # 各阶段 index、WAV、报告
  reports/audit_sep_run.json         # extract-sep 严格审计
  reports/kws_input_audit.json       # 下游输入契约审计（若 /root/kws 已存在）
  reports/all_stage_comparison.*     # 独立阶段比较
  reports/same_uid_audio_rank*       # 同 UID 音频去重/冲突分析
  best_sep/index.jsonl
  best_sep/{pos,neg}/{uid}.wav
  kws_handoff.json
```

不要只交付 `best_sep/*.wav`；index、handoff、审计报告和运行配置共同构成可复现证据。
