# extract `sep` — KWS 分离实验专用分支

仓库：https://github.com/powerli2024/extract.git  

**本目录只做 KWS BSS + Qwen CER oracle 选轨。不做 Presence / mix ASR。**  
竞赛 VE 用旁边的 **`/root/extract`**（分支 [`main`](https://github.com/powerli2024/extract/tree/main) 的 `ve/`）。不要在本克隆里 `git checkout main`。

下游选轨 / 条件 SE：[powerli2024/kws](https://github.com/powerli2024/kws.git)（读 `$VM_OUT`）。

```text
datasetA kws
  → extract-sep  ./run_sep.sh          # s1–s8，无 MMS-FA
  → $VM_OUT/{pos,neg}/s*/index.jsonl
  → $VM_OUT/best_sep/ + kws_handoff.json
  → kws  rebuild_best_sep / T0–T4
  → /root/extract/ve  用同一份 enroll + cmd 做拒识 / mix ASR
```

契约见 `KWS_HANDOFF.md`。选轨：**阶段内** argmin CER，平局保 `original`；**跨阶段** argmin CER，平局优先非 original。

### 完整时长与速度策略

- `--max-sep-sec` 仅为旧命令兼容参数，已不再截断；正式分离始终传入完整波形。
- `VM_SEP_BATCH` 控制并行批量（ONNX 默认 8）；全量待办先按时长排序，再组成满批，兼顾多 Session 利用率与长音频显存稳定性。
- 批量 OOM 自动二分降批；单条会清显存后按完整波形再试一次。仍失败则明确记为 error，由严格审计阻止不完整结果进入排行。
- 严格审计检查每条输出流与源音频的时长差；超过 20 ms 即失败，防止静默截断。

## AutoDL：两克隆并存

```text
/root/extract          # main：Presence + mix（不要 checkout sep）
/root/extract-sep      # 本仓 sep：KWS BSS
/root/miniconda3/envs/ve   # 共用 Python

/root/autodl-tmp/                    # 共用数据/权重
  datasetA/
  checkpoints/MossFormer2_ONNX/
  checkpoints/MossFormer2_SS_16K/
  Qwen3-ASR-1.7B/
  cache/{huggingface,torch,pip}/
  kws_sep/                           # 本仓 VM_OUT
  pos_neg/best_sep/                  # 给 main VE 的 enroll（可链到 kws_sep/best_sep）
  ve_mix_novad/
```

```bash
cd /root
git clone -b sep https://github.com/powerli2024/extract.git extract-sep
cd /root/extract-sep && chmod +x *.sh run_sep.sh pick_python.sh
export DATA_DIR=/root/autodl-tmp/datasetA
export VM_OUT=/root/autodl-tmp/kws_sep
export MOSS_CKPT_DIR=/root/autodl-tmp/checkpoints
export ASR_MODEL_DIR=/root/autodl-tmp/Qwen3-ASR-1.7B
# 已有 /root/extract 的 ve 环境：conda activate ve 即可
# 否则 ./setup_env.sh 会 conda create -n ve 并装 torch / ORT / qwen-asr / clearvoice
./setup_env.sh
source ./env.sh                 # 或: conda activate ve && source ve/.env_ve
./download_models.sh
./check_env.sh
./run_sep.sh --limit 20
./run_sep.sh

# 给 main VE 用同一份 enroll：
mkdir -p /root/autodl-tmp/pos_neg
ln -sfn /root/autodl-tmp/kws_sep/best_sep /root/autodl-tmp/pos_neg/best_sep
```

### 全量重跑并充分审计（推荐）

KWS 中文指标固定为**无调拼音 CER**，英文为字符 CER；不采用比赛严格中文字符 CER。
短唤醒词的拼音 CER 很离散，若 P50/P75/P90 选择相同 UID 集合，门控只运行首个
`thr_a/b/c`，其余在 summary 中写为 `duplicate_of`，不会重复分离和 ASR。

不做小规模测试，直接使用新的输出树全量运行：

```bash
cd /root/extract-sep
source ./env.sh
export VM_OUT=/root/autodl-tmp/kws_sep_dedup
export OLD_VM_OUT=/root/autodl-tmp/kws_sep
bash ./rerun_and_audit.sh --full
```

旧 `kws_sep` 不会被删除或混入新 index。若新目录中已有中断结果，只复用该新目录
内签名一致的成功 UID 并继续补齐。完成条件包括：1,838 UID、s1–s8 index/WAV
完整、中文 metric=pinyin、门控别名一致、best_sep WAV 完整；若 `/root/kws`
已同步，还会自动生成全阶段去重比较、同 ID 音频排行及旧/新 best_sep 对照。

然后 kws：

```bash
python scripts/rebuild_best_sep.py --pos-neg /root/autodl-tmp/kws_sep
python scripts/analyze_dual_zero.py --pos-neg /root/autodl-tmp/kws_sep
```

`./ve.sh` 在本克隆会直接退出；请 `cd /root/extract && ./ve.sh submit`。

## 环境与权重

与 `/root/extract` **同一套 conda `ve`**：数据、ONNX/ClearVoice 权重、Qwen3 只放 `/root/autodl-tmp`。  
`./setup_env.sh` 把 torch / ORT / qwen-asr / **clearvoice** 都装进 `ve`。

| 阶段 | 含义 |
|------|------|
| s1 | ONNX 一阶 original/spk1/spk2 |
| s2 | ClearVoice 一阶（同一 VE python） |
| s3 | ONNX cascade（输入同 split s1 wav） |
| s4 | ClearVoice cascade（输入 s2） |
| s5–s8 | 门控二次（抢救，不是默认 enroll） |
| handoff | 写出 `best_sep/` + `kws_handoff.json` |
