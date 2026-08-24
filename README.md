# extract `main` — Presence + mix ASR

仓库：https://github.com/powerli2024/extract.git  

**本目录只做竞赛拒识与 CMD 提取。** 不要在本克隆里 `git checkout sep`。  
KWS 八阶段分离用旁边的 **`/root/extract-sep`**（分支 [`sep`](https://github.com/powerli2024/extract/tree/sep)）。  
enroll 提纯：[powerli2024/kws](https://github.com/powerli2024/kws.git)。

```text
enroll = pos_neg/best_sep/{pos,neg}/{uid}.wav
cmd    = datasetA/{pos,neg}/cmd_*.wav
         │
         ▼
extract/ve   Presence（eres2netv2 + CMD 一次 ONNX 分离打分）
         ├─ 人不在 → 拒识
         └─ 人在   → 整段 mix ASR
```

Python **共用 conda 环境 `ve`**（与 `extract-sep` 同一解释器）。不要再建 `qwen3-asr` 或 `ClearerVoice-Studio`。

## AutoDL：两克隆并存

代码分两个文件夹；**数据、权重、缓存只放数据盘**，两边共用。

```text
/root/extract          # 本仓 main：Presence + mix
/root/extract-sep      # 另一克隆，branch sep：KWS BSS
/root/miniconda3/envs/ve

/root/autodl-tmp/                    # 共享
  datasetA/
  checkpoints/MossFormer2_ONNX/      # Presence USE_SEP + sep s1
  checkpoints/MossFormer2_SS_16K/    # 仅 extract-sep 的 ClearVoice 阶段
  Qwen3-ASR-1.7B/
  ve_models/
  cache/{huggingface,torch,pip,modelscope}/
  pos_neg/best_sep/                  # VE enroll（可链到 kws_sep/best_sep）
  kws_sep/                           # extract-sep 的 VM_OUT
  ve_mix_novad/                      # 本仓 VE 产物
```

```bash
# --- 本仓（main），不要 checkout sep ---
cd /root
git clone https://github.com/powerli2024/extract.git extract
cd /root/extract && git checkout main && git pull --ff-only
chmod +x ve.sh ve/*.sh
cd ve
cp -n .env_ve.example .env_ve
./setup_env.sh
conda activate ve
source .env_ve
ONLY=eres2netv2 ./download_presence_encoders.sh
./download_moss_onnx.sh
./download_qwen3_asr.sh
PIPELINE=mix ./check_env.sh

VE_OUT=/root/autodl-tmp/ve_mix_novad ./run_next_lift.sh submit

# --- 旁边的分离仓（另开目录，共享上面 autodl-tmp）---
cd /root
git clone -b sep https://github.com/powerli2024/extract.git extract-sep
# 然后: cd /root/extract-sep && ./setup_env.sh && ./run_sep.sh
# enroll 对齐:
#   ln -sfn /root/autodl-tmp/kws_sep/best_sep /root/autodl-tmp/pos_neg/best_sep
```

锁定 τ：zh `0.29305` / en `0.357868`。产物：`$VE_OUT/reports/submit/result.json`。禁止 `FORCE_CALIB`。

根目录 `./ve.sh submit` 等同 `cd ve && ./run_next_lift.sh submit`。

细节见 [`ve/CURRENT.md`](ve/CURRENT.md)、[`ve/SETUP.md`](ve/SETUP.md)。

注册音频上线前质检（不读取 CMD 或正负标签）：见
[`ve/ENROLL_QUALITY.md`](ve/ENROLL_QUALITY.md)，入口为
`ve/run_enroll_quality.sh`。
