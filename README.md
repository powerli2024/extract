# VM — 全量分离 8 组实验（pos / neg 分树）

**自包含流水线**：分离 / ASR 后端代码已整合进 `VM/scripts/`，AutoDL 上只需部署本目录 + 数据 + 权重，**不必再旁路挂载 VB / VB_onnx**。

代码仓库：https://github.com/powerli2024/extract.git

## 部署（Git clone → AutoDL）

系统盘仅约 30G：**数据、权重、输出必须放 `/root/autodl-tmp`**。代码本身很小，可放 `/root/extract`。

```bash
# 1) 拉代码
cd /root
git clone https://github.com/powerli2024/extract.git extract
cd /root/extract && chmod +x *.sh

# 2) 数据盘路径（有 autodl-tmp 时脚本会自动用这些默认值）
export DATA_DIR=/root/autodl-tmp/datasetA
export VM_OUT=/root/autodl-tmp/vm
export MOSS_CKPT_DIR=/root/autodl-tmp/checkpoints
export ASR_MODEL_DIR=/root/autodl-tmp/Qwen3-ASR-1.7B
export HF_HOME=/root/autodl-tmp/cache/huggingface
export TORCH_HOME=/root/autodl-tmp/cache/torch
export PIP_CACHE_DIR=/root/autodl-tmp/cache/pip

# 3) Python 依赖（默认清华 PyPI；torch 若无 CUDA 再回退官方 cu124）
./setup_env.sh
source ./env.sh

# 4) 权重：只跑这一个入口（不要单独跑 download_mossformer2_*.sh）
./download_models.sh            # ONNX + ClearVoice .pt → $MOSS_CKPT_DIR
# ./download_models.sh --asr    # 可选，拉 Qwen3-ASR-1.7B（很大）

# 5) 只读检查后试跑
./check_env.sh
./run_all.sh --limit 20
./run_all.sh
```

日常更新代码（不碰数据/权重）：

```bash
cd /root/extract   # 若 clone 到 VM 则 cd /root/VM
chmod +x update.sh && ./update.sh
# 若仍提示 would be overwritten: ./update.sh --hard
./setup_env.sh
```

`git pull` 若报 `Your local changes ... setup_env.sh would be overwritten`，说明本地改过脚本、**新版拉不下来**，跑的仍是旧逻辑（例如先打阿里云 pytorch-wheels）。应先更新代码，不要在这种状态下继续 `./setup_env.sh`。

| 必须自备 | 路径示例 | 说明 |
|----------|----------|------|
| `datasetA/{pos,neg}.jsonl` + `{pos,neg}/kws_*.wav` | `$DATA_DIR` | 本仓不包含数据 |
| Qwen3-ASR-1.7B | `$ASR_MODEL_DIR` | 体积大，不自动下载 |
| MossFormer2 ONNX / ClearVoice `.pt` | `$MOSS_CKPT_DIR` | `./download_models.sh` |
| GPU + conda | AutoDL 镜像自带 | Python 3.12 环境名 `qwen3-asr` |

ClearVoice 阶段（s2/s4/s5/s7/s8）额外：

```bash
# 推荐装进同一 qwen3-asr，避免双环境
conda activate qwen3-asr && pip install -r requirements-optional.txt
# 或独立环境:
# conda create -n ClearerVoice-Studio python=3.10 -y
# conda activate ClearerVoice-Studio && pip install clearvoice torch torchaudio
export CLEARVOICE_PYTHON=$CONDA_PREFIX/bin/python
```

## 权重（只认这一套路径）

AutoDL 上全部落在数据盘，**不要写 `/root/checkpoints`**（系统盘会满）。

```text
/root/autodl-tmp/checkpoints/MossFormer2_ONNX/simple_model.onnx     # s1/s3/s6
/root/autodl-tmp/checkpoints/MossFormer2_SS_16K/last_best_checkpoint.pt  # s2/s4/s5/s7/s8
/root/autodl-tmp/Qwen3-ASR-1.7B/                                    # ASR，加 --asr 才下载
```

| 做什么 | 命令 |
|--------|------|
| 下载分离权重 | `./download_models.sh` |
| 只要 ONNX | `./download_models.sh --onnx-only` |
| 只要 ClearVoice | `./download_models.sh --ss-only` |
| 额外拉 ASR | `./download_models.sh --asr` |
| 检查是否齐全 | `./check_env.sh` |

旧脚本若已经下到 `/root/checkpoints`，迁到数据盘：

```bash
cd /root/extract
bash migrate_to_autodl_tmp.sh
source /root/autodl-tmp/env_paths.sh
```

`download_mossformer2_ss.sh` / `download_mossformer2_onnx.sh` 是内部脚本，不要单独跑。

## 脚本分层

| 脚本 | 职责 |
|------|------|
| `check_env.sh` | **只读检查**（不安装、不下载） |
| `setup_env.sh` | **搭建**：pip、目录、环境快照 |
| `download_models.sh` | **唯一下载入口**（调用本包 `download_mossformer2_*.sh`） |
| `run_stage.sh` / `run_all.sh` | **只跑实验**：默认离线，缺权重报错 |

## 约定

| 项 | 规则 |
|----|------|
| uid | **`{split}_{id}`**（jsonl 的 `id`；禁止行号） |
| 数据 | `datasetA/{pos,neg}.jsonl`（id 唯一且与 `kws_{id}.wav` 一致） |
| 输出树 | `$VM_OUT/pos/...` 与 `$VM_OUT/neg/...` **互不混写** |
| 权重 | 仅 `./download_models.sh` 可下载；`run_*` 只读本地 |
| CER | CJK=无调拼音；英文=字符；神谕=候选最低 CER |
| 默认 splits | `pos,neg` |
| 阈值 a/b/c | 该 split 第一轮神谕 CER 的 P50/P75/P90 |

## 实验

| 阶段 | 含义 |
|------|------|
| s1 | ONNX 全量一阶 |
| s2 | ClearVoice 全量一阶 |
| s3 | ONNX cascade（复用同 split s1） |
| s4 | ClearVoice cascade（复用同 split s2） |
| s5 | s1 分布门控 → ClearVoice 一阶（peak） |
| s6 | s1 门控 → ONNX 二阶 |
| s7 | s2 门控 → ONNX 二阶 |
| s8 | s2 门控 → ClearVoice 二阶 |

## AutoDL

**大文件必须在数据盘** `/root/autodl-tmp`（系统盘仅 30G）。  
流水线默认：`VM_OUT` / `MOSS_CKPT_DIR` / 缓存 → 数据盘；若模型还在 `/root/...`，先迁移：

```bash
cd /root/VM && chmod +x *.sh
bash migrate_to_autodl_tmp.sh --dry-run   # 先看会迁什么
bash migrate_to_autodl_tmp.sh             # 迁模型/数据，原路径留 symlink
bash migrate_to_autodl_tmp.sh --also-caches
source /root/autodl-tmp/env_paths.sh
df -h / /root/autodl-tmp
```

```bash
# 只需上传/放置 VM/ 本身（不必再放 VB、VB_onnx）
cd /root/VM && chmod +x *.sh

./setup_env.sh                 # 创建 qwen3-asr + torch/torchaudio/onnxruntime-gpu/qwen-asr
./download_models.sh
./check_env.sh

source ./env.sh                # 或: conda activate qwen3-asr
# 下列路径在 AutoDL 上默认已是 autodl-tmp；显式写出亦可
export DATA_DIR=/root/autodl-tmp/datasetA
export VM_OUT=/root/autodl-tmp/vm
export MOSS_CKPT_DIR=/root/autodl-tmp/checkpoints
export ASR_MODEL_DIR=/root/autodl-tmp/Qwen3-ASR-1.7B   # 或仍指向 /root 下的 symlink

./run_all.sh --limit 20
./run_all.sh

# 从 s2 续跑（默认跳过已完成阶段，不覆盖 s1 等）
./run_all.sh --stages s2,s3,s4,s5,s6,s7,s8,compare,eval

# ClearVoice（s2/s4/s5/s7/s8 需要）
conda create -n ClearerVoice-Studio python=3.10 -y
conda activate ClearerVoice-Studio && pip install clearvoice
export CLEARVOICE_PYTHON=$CONDA_PREFIX/bin/python
# 或在 qwen3-asr 内: pip install clearvoice

# 强制重跑某阶段: --force 或 VM_FORCE=1
# ONNX 吞吐: MOSS_NUM_SESSIONS=6 VM_SEP_BATCH=8（run_* 默认已设）

# 结果分析（只读；完备性 / CER 排行 / 增益 / 失败补跑建议）
./run_stage.sh analyze
# → $VM_OUT/reports/analysis.md 与 analysis.json
# 细分布仍用: ./run_stage.sh eval
```

`setup_env.sh` 会安装：`torch` `torchaudio` `onnxruntime-gpu` `nvidia-*-cu12`，以及 **`requirements.txt`**。  
**所有 pip 默认走清华源**（`pip.conf` / `PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`）。  
CUDA 标签按 `nvidia-smi` 选 `cu124/cu121`；可覆盖：`TORCH_CUDA=cu124 ./setup_env.sh`。  
已完成阶段默认跳过（`VM_SKIP_DONE=1`）。

### pip / torch 源

1. 普通包、以及 **第一次装 torch**：清华 `pypi.tuna.tsinghua.edu.cn`
2. 仅当清华轮子 **没有 CUDA**（`torch.cuda.is_available()=False`）时，才回退 `https://download.pytorch.org/whl/cu124`
3. 不要用阿里云 `pytorch-wheels/cu124` 当 `--index-url`（经常 `from versions: none`）

```bash
# 已装到一半可直接重跑（缺啥补啥）
cd /root/VM && ./setup_env.sh

# 清华源已是 CPU 版、强制官方 CUDA wheel：
source /etc/network_turbo
TORCH_INDEX=https://download.pytorch.org/whl/cu124 ./setup_env.sh
```

`Running pip as the 'root' user` 在 AutoDL 上可忽略。


## 包内已整合的代码

```text
VM/scripts/
  utils_audio.py          # 音频 IO
  cer_metrics.py          # ASR 文本归一化
  asr_backend.py          # Qwen3-ASR
  mossformer2_ss.py       # ClearVoice MossFormer2
  mossformer2_onnx.py     # ONNX MossFormer2
  stage_*.py / collect…   # 本实验编排
VM/download_mossformer2_ss.sh
VM/download_mossformer2_onnx.sh
```

## 仍需环境侧提供（不是另一份代码仓）

| 项 | 说明 |
|----|------|
| 数据集 | `DATA_DIR`（如 `/root/datasetA`） |
| 分离权重 | `MOSS_CKPT_DIR` 下 ONNX / ClearVoice `.pt`（`download_models.sh`） |
| ASR 权重 | `ASR_MODEL_DIR`（需自行放置，体积大） |
| ClearVoice 运行时 | 本机 `CLEARVOICE_PYTHON` 环境（pip 包 `clearvoice`，非 VB 源码） |
| conda | `qwen3-asr`（跑 ASR + 主流程） |

## 产出树

```text
$VM_OUT/
  meta/collect_summary.json
  pos/{meta,s1_…,s8_…,reports}/
  neg/{meta,s1_…,s8_…,reports}/
  reports/{compare_all,eval_report,analysis}.*
  packs/vm_*.tar.gz
```
