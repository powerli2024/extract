# DAE-TSE 可选实验接入

## 1. 定位

`voice-interaction-challengecup` 中的 DAE-TSE 是中文文本条件目标说话人提取模型，不是普通两路盲分离器。它的输入是：

```text
16 kHz 单声道波形 + 已知唤醒文本的音素 cue → 一条目标语音
```

交接包提供了正式调用逻辑和辅助源码，但不包含 DAE-TSE 主仓、WeSep 运行环境、recipe 配置或模型权重。已记录的上游版本为 `GnafiY/DAE-TSE@b306b2ac70e33a95047f9366cc9bb5fde29c0de6`，中文权重 `model.pt` 约 221 MB。

因此本仓采用“单环境、候选流程隔离”：

- `ve-cu124` Python 同时负责 source 解析、DAE 推理、SHA256 签名、输出审计、Qwen ASR/CER 与报告；
- `PYTHON_BIN`、`CLEARVOICE_PYTHON`、`DAE_PYTHON_BIN` 默认是同一个解释器；
- DAE 输入保持完整时长并按正式链约定等比例归一到峰值 0.70；
- 输出固定放在 `$VM_OUT/experiments/`，现有 handoff 不会发现它；
- 报告始终写 `production_approved=false`，直到声纹和下游严格评测通过。

统一环境的版本清单、构建命令和验收见 [environment/README_CUDA124.md](environment/README_CUDA124.md)。DAE-TSE 官方环境为 PyTorch 2.9.1+cu126；本项目为满足 CUDA 12.4 使用 2.6.0+cu124，必须通过真实 checkpoint smoke 后才允许进入效果评测。

## 2. 代码结构

| 文件 | 职责 |
|---|---|
| `scripts/candidate_sources.py` | 解析 raw/s1–s8 的冻结 source、音频 hash 和实验签名 |
| `scripts/dae_tse_infer_manifest.py` | 在 DAE 专用环境中构建模型、文本转 cue、逐条推理 |
| `scripts/stage_dae_tse.py` | 生成 manifest、调用 DAE、严格检查时长、统一重算 source/DAE CER |
| `run_dae_tse.sh` | AutoDL 入口；不进入 `run_sep.sh` 默认阶段 |
| `scripts/test_dae_tse_integration.py` | 无权重的 source、subset 和签名回归测试 |

## 3. 必需资产

```text
/root/miniconda3/envs/ve-cu124/bin/python
/root/autodl-tmp/projects/DAE-TSE/
  examples/librimix/dae-tse/exp/backbone/config.yaml
  tools/build_zh_text_cues.py
/root/autodl-fs/midea-dae/models/dae_zh_v1/model.pt
```

统一 Python 必须能导入 `torch`、`yaml`、`soundfile`、`wesep`、`kce` 和 DAE 仓内的 `tools.build_zh_text_cues`。构建入口固定为 `bash ./setup_env.sh`，不要再维护独立 `dae-tse` 或 `qwen3-asr` 环境。

`DAE_TSE_REPO` 必须保留 Git 元数据；commit、tracked dirty diff、DAE Python 版本、配置和权重 SHA256 都会进入实验签名。若只有无 `.git` 的源码副本，应先恢复为固定 commit 的克隆，而不是关闭签名检查。

预检：

```bash
source /root/extract-sep/env.sh
"$PYTHON_BIN" - <<'PY'
import soundfile, torch, yaml, wesep
from tools.build_zh_text_cues import text_to_phone_labels
print("DAE runtime imports OK", torch.cuda.is_available())
PY
```

执行预检前应 `cd /root/autodl-tmp/projects/DAE-TSE`，确保本仓 `tools` 可导入。

## 4. 推荐实验顺序

先保证 s1 完整，再以 s1 的阶段内 winner 作为 source：

```bash
cd /root/extract-sep
source ./env.sh

export VM_OUT=/root/autodl-tmp/kws_sep_dedup
export DAE_PYTHON_BIN="$PYTHON_BIN"
export DAE_TSE_REPO=/root/autodl-tmp/projects/DAE-TSE
export DAE_TSE_CHECKPOINT=/root/autodl-fs/midea-dae/models/dae_zh_v1/model.pt
export DAE_SOURCE_STAGE=s1
export DAE_OUT=/root/autodl-tmp/kws_sep_dae_s1_v1

# 20 条/每 split：先验证真实模型输出，不跑 ASR
LIMIT=20 bash ./run_dae_tse.sh --skip-asr

# 正式全量必须换新目录，不能复用 smoke 签名
export DAE_OUT=/root/autodl-tmp/kws_sep_dae_s1_full_v1
LIMIT=0 bash ./run_dae_tse.sh
```

查看：

```text
$DAE_OUT/signature.json
$DAE_OUT/dae_manifest.jsonl
$DAE_OUT/dae_runtime_results.jsonl
$DAE_OUT/index.jsonl
$DAE_OUT/summary.json
$DAE_OUT/wav/{pos,neg}/*_dae_tse.wav
```

对 s7 门控 cohort 单独评估：

```bash
export DAE_SOURCE_STAGE=s7
export DAE_SOURCE_THR=a
export DAE_OUT=/root/autodl-tmp/kws_sep_dae_s7a_full_v1
LIMIT=0 bash ./run_dae_tse.sh
```

s7 本身依赖 s2；`source_thr` 必须与已审计的 s7 arm 一致。这个命令只评估该门控 cohort，不会把未触发 UID 静默补成 s7。

## 5. 当前选择规则

对中文适用样本，source 和 DAE 输出使用同一个 Qwen 实例重新识别：

```text
DAE CER < source CER  → 实验 winner=dae_tse
否则                 → winner=source（平局不切换）
```

英文当前标记为 `fallback_source_unsupported_language`，不送入中文 DAE。这样 index 仍解释全部 source population，但 DAE 效果统计只计算中文 paired subset。

此 winner 仍是文本 oracle，仅用于判断 DAE 值不值得继续，不允许直接复制进 `best_sep`。

## 6. Go/No-Go

第一层，运行契约：

- signature 与数据、source WAV、配置、权重、ASR 目录一致；
- 所有中文适用 UID 均有 16 kHz、精确同长度、finite 的 DAE WAV；
- source/DAE 使用同一轮 ASR，paired coverage 100%；
- 任一失败、签名冲突或缺失都停止排行。

第二层，候选筛选：

- 查看 `n_dae_strict_improve`、`n_dae_worse`、两臂 mean CER 和 CER=0 保持率；
- DAE 只在严格改善时切换，平局保留 source；
- 后续加入 `cos(embed(DAE), embed(source)) >= 0.92` 和 `CER_DAE <= CER_source + 0.05` 双安全门；cosine 只作灾变门，不作纯净度主排序。

第三层，生产批准：

- 在 `/root/kws` 中与冻结 s1→s7 路由组合并全量复算；
- 在 `/root/extract@main` 运行冻结 Presence、CMD FRR/FAR 和严格比赛评测；
- 保证 finite-CER、paired 和 UID 覆盖与基线完全相等；
- 未取得稳定增益时保持 `NO_GO`。

## 7. 与正式交接包的差异

`voice-interaction-challengecup/formal_v21` 的完整链还包含 MMS-FA 裁切、DAE 后 MossFormer2_SE_48K、WavLM 特征和冻结 V7 gate。本次先隔离评估 DAE-TSE 本体，原因是：

1. extract-sep 当前坚持完整 KWS 波形，不应在未做对照前把 MMS-FA 裁切混进同一实验；
2. 同时引入裁切、DAE、SE 和 gate 无法判断增益来自哪一项；
3. 正式 V7 阈值是在另一套候选分布上冻结的，不能直接迁移。

若 DAE 本体过第一、二层门，再分别增加 `MMSFA crop + DAE` 和 `DAE + SE48K` 两个独立实验臂。
