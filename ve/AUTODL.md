# AutoDL 部署清单（4090 开发 / L20 比赛）

## 磁盘布局

```text
/root/autodl-tmp/                 # 数据盘：大文件只放这里
  datasetA/
  pos_neg/best_sep/               # 干净 KWS enroll
  ve_models/
    PS4/{checkpoint_epoch037.pt,inference.py}
    cnceleb_resnet34_LM/
    eres2netv2_zh/
    wesep/                        # PIPELINE=wesep（./download_wesep.sh）
  checkpoints/MossFormer2_ONNX/   # PIPELINE=sep_route（./download_moss_onnx.sh）
    simple_model.onnx
  ve_ps4_vad/ | ve_wesep_vad/ | ve_mix_vad/ | …_novad/
                              # 各方案 VE_OUT（按 PIPELINE + enroll VAD 分目录并存）
  ve_presence_best/reports/presence_calib_<backend>_<sep>_<ls>_<vad>_<norm>/
  cache/{huggingface,modelscope,torch,pip}/

/root/extract/ve/                   # 本包代码
/root/extract/scripts/           # MossFormer ONNX 封装（sep_route 必需）
```

## 三种方案（均可在 AutoDL 跑通）

| PIPELINE | TSE | 额外准备 | 默认输出 |
|----------|-----|----------|----------|
| `ps4` | HF PS4 BSRNN | `./download_models.sh` | `/root/autodl-tmp/ve_ps4_vad` |
| `wesep` | WeSep bsrnn_ecapa_vox1 | `./download_wesep.sh` | `/root/autodl-tmp/ve_wesep_vad` |
| `sep_route` | MossFormer 分离 + 声纹选路 | `./download_moss_onnx.sh`（同仓 `../scripts`） | `/root/autodl-tmp/ve_sep_route_vad` |
| `mix` | CMD 直通 | — | `/root/autodl-tmp/ve_mix_vad` |

`ENROLL_VAD=0` 时后缀为 `_novad`，与 VAD 结果并存。校准 thr 亦分桶；`run_extract` 校验 `enroll_vad` 一致。  
建议用 `HOLDOUT_FRAC=0.3` 选 thr，关注 `holdout.contest_score`，勿迷信同集最优 thr。

```bash
cd /root/extract/ve
chmod +x *.sh
conda activate ve
./setup_env.sh && source .env_ve
./download_models.sh
./check_env.sh                         # 或 PIPELINE=wesep ./check_env.sh

# A) PS4
PIPELINE=ps4 ./run_all.sh

# B) WeSep
./download_wesep.sh
PIPELINE=wesep ./run_all.sh

# C) MossFormer 选路
# 需: /root/extract/scripts 与 ONNX
./download_moss_onnx.sh
PIPELINE=sep_route ./run_all.sh

# 冒烟
LIMIT=32 SKIP_ASR=1 PIPELINE=ps4 ./run_all.sh
```

显式指定目录：`VE_OUT=/root/autodl-tmp/ve_ps4 PIPELINE=ps4 ./run_all.sh`  
（若 `VE_OUT` 仍是旧的 `/root/autodl-tmp/ve`，`run_all` 会改写为 `ve_${PIPELINE}`。）

## datasetA / best_sep

| 数据 | Windows 源 | AutoDL |
|------|------------|--------|
| CMD | `d:\media\datasetA` | `/root/autodl-tmp/datasetA` |
| enroll | `d:\media\pos_neg\best_sep` | `/root/autodl-tmp/pos_neg/best_sep` |

## 换行符

`VE/*.sh` 须为 **LF**。若 `$'\r': command not found`：

```bash
sed -i 's/\r$//' /root/extract/ve/*.sh
```

## 方案依赖速查

| 组件 | ps4 | wesep | sep_route |
|------|-----|-------|-----------|
| PS4 checkpoint + inference.py | 必需 | — | — |
| ERes2Net / cnceleb | 必需 | 必需 | 必需 |
| wenet-e2e/wesep + ModelScope 权重 | — | 必需 | — |
| MossFormer ONNX + `extract/scripts/mossformer2_onnx.py` | — | — | 必需 |
| Qwen3-ASR（`download_qwen3_asr.sh`） | 算 CER 时 | 同左 | 同左 |

`sep_route` 会强制 `USE_SEP=1`（Presence 对 mix/spk1/spk2 取 max sim），阈值单独校准，勿与 ps4 共用 thr。

## GPU / 显存

| 卡 | 建议 |
|----|------|
| 4090 24GB | `DEVICE=cuda:0`，TSE batch=1 |
| L20 | 同左 |

- PS4 OOM：`tse_ps4.py` 缩短 CMD（6→2s）重试  
- neg 门控拒识后不跑 TSE  
- MossFormer：优先 CUDA EP；失败会提示装 `onnxruntime-gpu`

## 报告

```text
$VE_OUT/manifest/
$VE_OUT/reports/presence_calib/
$VE_OUT/results/{pos,neg,all}_results.jsonl
$VE_OUT/extracted/{pos,neg}/*.wav
$VE_OUT/reports/{summary.*,asr_cer/}
$VE_OUT/logs/run_*.log
```

竞赛分 `0.5*RR + 0.5*(1-CER)`：accept **必须**跑 ASR；未跑时 CER/contest 为 `null`。

## 常见坑

1. 大数据不在 autodl-tmp → 软链或改 `DATA_DIR` / `BEST_SEP_DIR`  
2. HF 超时 → `source /etc/network_turbo`；`HF_ENDPOINT=https://hf-mirror.com`  
3. ERes2Net 失败 → `PRESENCE_BACKEND=resnet34`  
4. `PIPELINE=wesep` 报 import → `./download_wesep.sh` 并 `export WESEP_ROOT=...`  
5. `PIPELINE=sep_route` → `./download_moss_onnx.sh` + 同步 `VM/`  
6. shell CRLF → `sed -i 's/\r$//' *.sh`  
