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
