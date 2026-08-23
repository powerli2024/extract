# VE：Presence-gated Target Speaker Extraction

**当前提交默认是 Presence + mix ASR，不是 ps4。** 环境：conda `ve`（[`SETUP.md`](SETUP.md)）。最优流水线见 [`CURRENT.md`](CURRENT.md)。

KWS 八阶段分离不在本分支：用 extract **`sep`** 写出 `best_sep`，本目录只读 enroll。路径用 `BEST_SEP_DIR` 或 `--best-sep` 自行指定（例如 `/root/autodl-tmp/kws_sep/best_sep`）。

**Verify-then-Extract**：声纹判断 CMD 是否存在该说话人；**仅「人不在」时拒识**；在场则按所选方案提取目标语音。

**ASR 全量流探测** [`asr_probe/`](asr_probe/README.md)（pos+neg × mix/d1/d2，不做拒识）。

## 口径

| 项 | 约定 |
|----|------|
| Enroll | 正式运行默认 `STRICT_ENROLL=1`：每个 UID 必须有 `best_sep` 的 `ok=true` 注册波，绝不静默回退原始 KWS |
| CMD | `datasetA/{pos,neg}/cmd_*.wav` |
| 标签 | pos=`present`，neg=`absent` |
| 拒识 | **仅** `presence_score < thr` → `reject_reason=speaker_absent` |
| CER | 官方 `CER=(S+I+D)/N`；不截断。误拒/失败 pos 令该条 `CER=1` |
| 竞赛分 | `0.5*RR_neg + 0.5*(1-CER_pos_micro)`；CER 按总错误/总参考字符计算 |

```text
enroll(best_sep) + dirty CMD → PresenceGate
                                ├─ target absent → REJECT
                                └─ target present → route/TSE → ASR → optional risk reject

标签和 `cmd_text` 只供离线最终计分，**不得**参与线上候选选择、Presence 或 ASR 前处理。
```

## 三种 AutoDL 方案

| PIPELINE | 提取 | 准备 |
|----------|------|------|
| `ps4`（默认） | HF [PS4](https://huggingface.co/TaurenMountain/PS4) | `./download_models.sh` |
| `wesep` | WeSep `bsrnn_ecapa_vox1` | `./download_wesep.sh` |
| `sep_route` | MossFormer 分离 + enroll 选路 | `./download_moss_onnx.sh`（同仓 `../scripts`） |
| `adaptive_route` | 只有分离流声纹比分数更高至少 `ROUTE_MIN_GAIN` 才选分离流，否则保留 mix | 同上 |

默认输出按 **PIPELINE + enroll VAD** 分目录，可并存：

- `ENROLL_VAD=1` → `/root/autodl-tmp/ve_${PIPELINE}_vad`
- `ENROLL_VAD=0` → `/root/autodl-tmp/ve_${PIPELINE}_novad`

Presence 校准亦分桶：`ve_presence_best/reports/presence_calib_<backend>_<sep>_<ls>_<vad>_<norm>/`。  
`run_extract` 会校验 thr 文件的 `enroll_vad` 与当前开关一致，避免串用。
正式 `run_all.sh` 默认 `ASR_RESUME=0`，避免注册波、路由参数或提取音频变化后复用旧转写；只有确认输入音频与解码配置不变时才可显式设为 `ASR_RESUME=1`。

```bash
cd /root/extract/ve
./setup_env.sh && source .env_ve
./download_models.sh && ./check_env.sh

PIPELINE=ps4 ./run_all.sh
ENROLL_VAD=0 PIPELINE=mix ./run_all.sh          # 与 VAD 跑并存
HOLDOUT_FRAC=0.3 PIPELINE=mix ./run_all.sh      # thr 用 holdout 评估泛化
./download_wesep.sh && PIPELINE=wesep ./run_all.sh
./download_moss_onnx.sh && PIPELINE=sep_route ./run_all.sh
./download_moss_onnx.sh && PIPELINE=adaptive_route ROUTE_MIN_GAIN=0.03 ./run_all.sh

# 冒烟
LIMIT=32 SKIP_ASR=1 PIPELINE=ps4 ./run_all.sh
```

默认输出：`/root/autodl-tmp/ve_${PIPELINE}_vad`（或 `_novad`）。详情见 [`AUTODL.md`](AUTODL.md)。

## 分步命令

```bash
source .env_ve
export PYTHONPATH="$VE_ROOT/scripts:$VE_ROOT/../scripts:$PYTHONPATH"

python scripts/build_manifest.py --strict-best-sep
python scripts/calibrate_presence.py --target-frr 0.02
# sep_route 校准加 --use-sep

python scripts/run_extract.py \
  --thr-file $VE_OUT/reports/presence_calib/recommended_thr.json \
  --tse-backend ps4          # 或 wesep_bsrnn / sep_route / adaptive_route

./run_asr_cer.sh             # 真实 CER
python scripts/final_evaluate.py --ve-out "$VE_OUT" --strict
```

## 产物

```text
$VE_OUT/
  manifest/ samples.jsonl ...
  results/{pos,neg,all}_results.jsonl
  extracted/{pos,neg}/{uid}.wav
  reports/presence_calib/ summary.* asr_cer/ final_eval/
  logs/
```

## 与 VD 差异

VD 可用 RMS/ASR/SIM 多条件拒识；VE **只保留「人不在」**（`reject_policy=speaker_absent_only`）。

## 递进实验顺序

1. `PIPELINE=mix EXTRA_REJECT=0`：严格注册波 + 纯 Presence 基线；仅替换 KWS 提纯版本。
2. `PIPELINE=adaptive_route EXTRA_REJECT=0`：统一的、标签无关的 mix/分离流路由；调 `ROUTE_MIN_GAIN`，阈值需在独立开发集重新冻结。
3. `PIPELINE=sep_route`、PS4、WeSep、Cond-TasNet：作为候选目标提取臂，与阶段 2 做同 manifest 的配对比较。
4. `EXTRA_REJECT=1`：仅在独立开发集冻结文本/多编码器加拒，再在测试集报告为后置增强结果；不要用于归因 KWS 注册提纯。

每一臂都以 `reports/final_eval/summary.json` 为唯一最终口径，并保存 `best_sep/index.jsonl` 的 SHA-256。
`EXTRA_REJECT=0` 时最终评测显式使用本轮 `all_results.jsonl`；开启 overlay 时才显式使用本轮 `submit_rows.jsonl`，不会自动读取历史 overlay。
