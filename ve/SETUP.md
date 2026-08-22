# VE / VP 新环境清单

复制到 AutoDL 后按节执行。大文件只放 `/root/autodl-tmp/`。

---

## 1. 目录布局（必有 vs 跑出来的）

```text
/root/extract/ve/                          # 代码（git）
/root/extract/scripts/mossformer2_onnx.py   # USE_SEP=1 / sep_route / VP matrix 需要
# VD/tools 可选（ResNet 回退已拷进 VE/scripts）

/root/autodl-tmp/
  datasetA/                              # CMD：pos/neg + jsonl   【必拷】
  pos_neg/best_sep/{pos,neg}/*.wav       # enroll                 【必拷】
  ve_models/
    eres2netv2_zh/                       # Presence 默认
    campplus_zh/                         # VP 对照
    cnceleb_resnet34_LM/                 # 回退 / 对照
    PS4/{checkpoint_epoch037.pt,inference.py}   # 仅 PIPELINE=ps4
    wesep/                               # 仅 PIPELINE=wesep
  checkpoints/MossFormer2_ONNX/simple_model.onnx
  Qwen3-ASR-1.7B/                        # 真实 CER
  cache/{huggingface,modelscope,torch,pip}/
  clean_kws/  mix500/                    # 仅 AS-Norm，可暂不拷
  ve_gate_cmp/sep_streams/d1/            # VP MODE=matrix 可复用；没有就 MODE=live
```

Windows 对应：`d:\media\datasetA`、`d:\media\pos_neg\best_sep`。

---

## 2. 环境变量

新建 `ve/.env_ve`（不要提交 git）：

```bash
cp /root/extract/ve/.env_ve.example /root/extract/ve/.env_ve
# 按实际改 VE_ROOT 若代码不在 /root/extract/ve
```

### 2.1 路径（几乎必设）

| 变量 | 默认 | 作用 |
|------|------|------|
| `VE_ROOT` | `/root/extract/ve` | 代码根 |
| `PYTHONPATH` | `$VE_ROOT/scripts:$VE_ROOT/../scripts` | VE 脚本 + 同仓 Moss ONNX |
| `DATA_DIR` | `/root/autodl-tmp/datasetA` | CMD |
| `BEST_SEP_DIR` | `/root/autodl-tmp/pos_neg/best_sep` | enroll |
| `VE_MODEL_DIR` | `/root/autodl-tmp/ve_models` | 声纹 / PS4 / wesep |
| `VE_OUT` | 不设则 `run_all` → `ve_${PIPELINE}_{vad\|novad}` | 本轮产物 |
| `VE_OUT_BASE` | `/root/autodl-tmp/ve` | 旧兼容 |
| `CALIB_ROOT` | `/root/autodl-tmp/ve_presence_best` | 共享校准 |
| `VP_OUT` | `/root/autodl-tmp/vp` | 拒识模块产物 |
| `SAMPLES` | `$VE_OUT/manifest/samples.jsonl` | 可手指定 |
| `SEP_ROOT` | 自动找 `ve_gate_cmp/sep_streams` | 须含 `d1/` |
| `DEVICE` | `cuda:0` | |
| `PYTHON_BIN` | conda env `ve` 的 python | 与 ASR / ClearVoice 同一解释器；`./setup_env.sh` 会写入 `.env_ve` |

### 2.2 模型路径

| 变量 | 默认文件/目录 |
|------|----------------|
| `ERES2NET_DIR` / `ERES_DIR` | `$VE_MODEL_DIR/eres2netv2_zh` |
| `CAMPPLUS_DIR` | `$VE_MODEL_DIR/campplus_zh` |
| `SPK_CHS_DIR` | `$VE_MODEL_DIR/cnceleb_resnet34_LM` |
| `VBLINK_DIR` | `$VE_MODEL_DIR/vblink2_samresnet34`（可选） |
| `PS4_WEIGHTS` | `$VE_MODEL_DIR/PS4/checkpoint_epoch037.pt` |
| `WESEP_ROOT` | `$VE_MODEL_DIR/wesep` |
| `WESEP_MODEL_DIR` | WeSep 权重（download_wesep 写入） |
| `MOSS_ONNX_PATH` | `/root/autodl-tmp/checkpoints/MossFormer2_ONNX/simple_model.onnx` |
| `ASR_MODEL_DIR` / `QWEN3_ASR_DIR` | `/root/autodl-tmp/Qwen3-ASR-1.7B` |
| `COHORT_DIR` | `/root/autodl-tmp/clean_kws`（AS-Norm） |
| `TEST_COHORT_DIR` | `/root/autodl-tmp/mix500` |

### 2.3 缓存 / 网络 / 线程

| 变量 | 默认 |
|------|------|
| `HF_HOME` | `/root/autodl-tmp/cache/huggingface` |
| `MODELSCOPE_CACHE` | `/root/autodl-tmp/cache/modelscope` |
| `TORCH_HOME` | `/root/autodl-tmp/cache/torch` |
| `PIP_CACHE_DIR` | `/root/autodl-tmp/cache/pip` |
| `HF_ENDPOINT` | `https://hf-mirror.com` |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | `16`（非法值会导致 libgomp 报错） |

### 2.4 实验开关（`run_all.sh`，默认值）

| 变量 | 默认 | 含义 |
|------|------|------|
| `PIPELINE` | `ps4` | `mix` / `ps4` / `wesep` / `sep_route` |
| `PRESENCE_BACKEND` | `eres2netv2` | |
| `USE_SEP` | `1` | Presence 一次分离打分 |
| `LANG_SPLIT` | `1` | zh/en 分 τ |
| `ENROLL_VAD` | `1` | VP 建议对照时用 `0` |
| `HOLDOUT_FRAC` | `0` | 建议 `0.3` |
| `FORCE_CALIB` / `SKIP_CALIB` | `0` | |
| `LIMIT` | `0` | 冒烟用 `32`/`64` |
| `SKIP_ASR` | `0` | |
| `ASNORM` / `ENROLL_ZNORM` | `0` | 默认不要开 |

`./run_all.sh --help` 为完整列表。

---

## 3. 新建环境命令

```bash
# --- 代码（extract.git，VE 在 ve/）---
cd /root/extract
git pull --ff-only
chmod +x *.sh ve.sh ve/*.sh ve/VP/run_vp.sh
cd /root/extract/ve
cp -n .env_ve.example .env_ve
source .env_ve

# --- Python ---
./setup_env.sh
conda activate ve          # 若用了 conda 环境名 ve
source .env_ve             # setup 后再 source 一次

# --- 数据（从本机拷/解压到 autodl-tmp）---
# datasetA  → /root/autodl-tmp/datasetA
# best_sep  → /root/autodl-tmp/pos_neg/best_sep

# --- 权重 ---
./download_presence_encoders.sh          # eres + camp + resnet
./download_models.sh                     # PS4（仅要 ps4 时）
./download_moss_onnx.sh                  # Presence USE_SEP / sep_route / VP
./download_qwen3_asr.sh                  # CER
# ./download_wesep.sh                    # 仅 PIPELINE=wesep

./check_env.sh

# --- 自检路径 ---
ls "$DATA_DIR/pos" "$BEST_SEP_DIR/pos" | head
ls "$ERES2NET_DIR" "$MOSS_ONNX_PATH" "$ASR_MODEL_DIR" | cat
python -c "import mossformer2_onnx; print('moss ok')"
```

最小可跑（Presence + mix + ASR，推荐新环境第一条正式命令）：

```bash
cd /root/extract/ve && source .env_ve && conda activate ve
FORCE_CALIB=1 HOLDOUT_FRAC=0.3 ENROLL_VAD=0 PIPELINE=mix LIMIT=32 SKIP_ASR=1 ./run_all.sh
# 通过后再 LIMIT=0 且去掉 SKIP_ASR
```

拒识模块（不跑 ASR）：

```bash
cd /root/extract/ve/VP && source ../.env_ve
# 已有 sep_streams：
MODE=matrix ./run_vp.sh
# 否则现打：
MODE=live ENROLL_VAD=0 HOLDOUT_FRAC=0.3 ./run_vp.sh
```

---

## 4. 新环境自检清单

```bash
test -d /root/autodl-tmp/datasetA && echo OK_DATA
test -d /root/autodl-tmp/pos_neg/best_sep/pos && echo OK_ENROLL
test -d /root/autodl-tmp/ve_models/eres2netv2_zh && echo OK_ERES
test -f /root/autodl-tmp/checkpoints/MossFormer2_ONNX/simple_model.onnx && echo OK_MOSS
test -d /root/autodl-tmp/Qwen3-ASR-1.7B && echo OK_ASR
test -f /root/extract/scripts/mossformer2_onnx.py && echo OK_VM
```

缺哪项补哪项；不要把 `ve_mix_*` / `vp/` 跑数当「环境依赖」。
