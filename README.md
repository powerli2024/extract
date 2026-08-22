# extract `main` — Presence + mix ASR

仓库：https://github.com/powerli2024/extract.git  

**本分支只做竞赛拒识与 CMD 提取。** KWS 八阶段分离在 [`sep`](https://github.com/powerli2024/extract/tree/sep)。enroll 提纯在 [powerli2024/kws](https://github.com/powerli2024/kws.git)。

```text
enroll = pos_neg/best_sep/{pos,neg}/{uid}.wav
cmd    = datasetA/{pos,neg}/cmd_*.wav
         │
         ▼
extract/ve   Presence（eres2netv2 + CMD 一次 ONNX 分离打分）
         ├─ 人不在 → 拒识
         └─ 人在   → 整段 mix ASR
```

Python **全部进 conda 环境 `ve`**。不要再建 `qwen3-asr` 或 `ClearerVoice-Studio`。

## AutoDL

```bash
cd /root/extract
git checkout main && git pull --ff-only
chmod +x ve.sh ve/*.sh
cd ve
cp -n .env_ve.example .env_ve
./setup_env.sh
conda activate ve
source .env_ve
ONLY=eres2netv2 ./download_presence_encoders.sh
./download_moss_onnx.sh          # Presence USE_SEP=1
./download_qwen3_asr.sh          # 若还没有 Qwen3-ASR-1.7B
PIPELINE=mix ./check_env.sh

# 已有 ve_mix_novad：不要重跑 Presence / pos ASR
VE_OUT=/root/autodl-tmp/ve_mix_novad ./run_next_lift.sh submit

# 全新机器（禁止 FORCE_CALIB）
ENROLL_VAD=0 PIPELINE=mix \
PRESENCE_BACKEND=eres2netv2 USE_SEP=1 LANG_SPLIT=1 \
LOCKED_THR=1 EXTRA_REJECT=1 \
./run_all.sh
```

锁定 τ：zh `0.29305` / en `0.357868`。产物：`$VE_OUT/reports/submit/result.json`。

根目录 `./ve.sh submit` 等同 `cd ve && ./run_next_lift.sh submit`。

KWS 要重分离时：`git clone -b sep https://github.com/powerli2024/extract.git`，不要在本分支跑 s1–s8。

细节见 [`ve/CURRENT.md`](ve/CURRENT.md)、[`ve/SETUP.md`](ve/SETUP.md)。
