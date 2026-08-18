# VE：Presence-gated Target Speaker Extraction

**当前提交默认不是 ps4。** 环境构建见 [`SETUP.md`](SETUP.md)，最优流水线见 [`CURRENT.md`](CURRENT.md)（Presence + **mix** ASR）。

**Verify-then-Extract**：声纹判断 CMD 是否存在该说话人；**仅「人不在」时拒识**；在场则按所选方案提取目标语音。

**ASR 全量流探测** [`asr_probe/`](asr_probe/README.md)（pos+neg × mix/d1/d2，不做拒识）。

## 口径

| 项 | 约定 |
|----|------|
| Enroll | `pos_neg/best_sep/{split}/{uid}.wav`（干净 KWS） |
| CMD | `datasetA/{pos,neg}/cmd_*.wav` |
| 标签 | pos=`present`，neg=`absent` |
| 拒识 | **仅** `presence_score < thr` → `reject_reason=speaker_absent` |
| 竞赛分 | `0.5*RR + 0.5*(1-CER)`（accept 须真实 ASR CER） |

```text
enroll(best_sep) + cmd → PresenceGate
                           ├─ absent → REJECT
                           └─ present → TSE (ps4 | wesep | sep_route) → ASR CER
```

## 三种 AutoDL 方案

| PIPELINE | 提取 | 准备 |
|----------|------|------|
| `ps4`（默认） | HF [PS4](https://huggingface.co/TaurenMountain/PS4) | `./download_models.sh` |
| `wesep` | WeSep `bsrnn_ecapa_vox1` | `./download_wesep.sh` |
| `sep_route` | MossFormer 分离 + enroll 选路 | `./download_moss_onnx.sh`（同仓 `../scripts`） |

默认输出按 **PIPELINE + enroll VAD** 分目录，可并存：

- `ENROLL_VAD=1` → `/root/autodl-tmp/ve_${PIPELINE}_vad`
- `ENROLL_VAD=0` → `/root/autodl-tmp/ve_${PIPELINE}_novad`

Presence 校准亦分桶：`ve_presence_best/reports/presence_calib_<backend>_<sep>_<ls>_<vad>_<norm>/`。  
`run_extract` 会校验 thr 文件的 `enroll_vad` 与当前开关一致，避免串用。

```bash
cd /root/extract/ve
./setup_env.sh && source .env_ve
./download_models.sh && ./check_env.sh

PIPELINE=ps4 ./run_all.sh
ENROLL_VAD=0 PIPELINE=mix ./run_all.sh          # 与 VAD 跑并存
HOLDOUT_FRAC=0.3 PIPELINE=mix ./run_all.sh      # thr 用 holdout 评估泛化
./download_wesep.sh && PIPELINE=wesep ./run_all.sh
./download_moss_onnx.sh && PIPELINE=sep_route ./run_all.sh

# 冒烟
LIMIT=32 SKIP_ASR=1 PIPELINE=ps4 ./run_all.sh
```

默认输出：`/root/autodl-tmp/ve_${PIPELINE}_vad`（或 `_novad`）。详情见 [`AUTODL.md`](AUTODL.md)。

## 分步命令

```bash
source .env_ve
export PYTHONPATH="$VE_ROOT/scripts:$VE_ROOT/../scripts:$PYTHONPATH"

python scripts/build_manifest.py
python scripts/calibrate_presence.py --target-frr 0.02
# sep_route 校准加 --use-sep

python scripts/run_extract.py \
  --thr-file $VE_OUT/reports/presence_calib/recommended_thr.json \
  --tse-backend ps4          # 或 wesep_bsrnn / sep_route

./run_asr_cer.sh             # 真实 CER
```

## 产物

```text
$VE_OUT/
  manifest/ samples.jsonl ...
  results/{pos,neg,all}_results.jsonl
  extracted/{pos,neg}/{uid}.wav
  reports/presence_calib/ summary.* asr_cer/
  logs/
```

## 与 VD 差异

VD 可用 RMS/ASR/SIM 多条件拒识；VE **只保留「人不在」**（`reject_policy=speaker_absent_only`）。
