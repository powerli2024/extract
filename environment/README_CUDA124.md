# Ubuntu 20.04 + CUDA 12.4 统一环境清单

## 1. 已固定的口径

`CUDA 12.4` 在本项目中指 **PyTorch cu124 wheel 运行时**：CUDA 与 cuDNN 动态库由 PyTorch wheel 提供，ONNX Runtime 复用同一组库。默认不安装 `/usr/local/cuda-12.4`、`nvcc` 或系统级 cuDNN；当前流程只做推理，不编译 CUDA 扩展。

若后续要编译自定义 CUDA 算子，再单独安装 CUDA Toolkit 12.4.1，并继续保持 Python 环境中的 cu124 版本不变。

## 2. 目标清单

| 层级 | 固定项 | 验收 |
|---|---|---|
| 系统 | Ubuntu 20.04 x86_64 | `/etc/os-release` 必须为 `ubuntu/20.04` |
| GPU 驱动 | wheel 运行时 `>=525.60.13`；Toolkit 12.4 GA `>=550.54.14` | `nvidia-smi` 可用且版本达标 |
| Python | Conda `ve-cu124`, Python 3.10.x | 不复用 Python 3.12 或旧 cu126 环境 |
| CUDA 主栈 | torch 2.6.0、torchaudio 2.6.0、torchvision 0.21.0 | 三者均来自官方 `cu124` 索引；`torch.version.cuda == '12.4'` |
| ONNX | onnxruntime-gpu 1.20.2 | 必须出现 `CUDAExecutionProvider` |
| 分离 | clearvoice 0.1.2 | `import clearvoice` 成功 |
| ASR | qwen-asr 0.0.6、transformers 4.57.6、accelerate 1.12.0 | `import qwen_asr` 成功 |
| Presence / 回退 | modelscope 1.39.1、funasr 1.4.6、kaldiio 2.18.1、addict 2.4.0 | 与主流程共用解释器 |
| DAE-TSE | `/root/DAE-TSE`（当前 `9799ec6...`）的 `wesep` 与 `kce` | 与主流程共用 `PYTHON_BIN`；中文 helper/资产另验 |
| 系统工具 | ffmpeg、sox、git、libsndfile、zlib | 安装脚本预检 |
| 可复现证据 | 完整包锁 + JSON 验证报告 | 每次构建后重新生成 |

依据：PyTorch 官方提供 2.6.0 的 cu124 组合；ONNX Runtime 1.20.x 对应 CUDA 12.x、cuDNN 9.x，并可复用 PyTorch 的 CUDA/cuDNN 库。NVIDIA CUDA 12.x minor compatibility 的 Linux driver 下限是 525.60.13；若安装完整 CUDA Toolkit 12.4 GA，则要求至少 550.54.14。

依赖冲突处理：不安装 `modelscope[audio]` extra。该 extra 在 1.39.1 中固定 `librosa==0.10.1`，与 ClearVoice 0.1.2 固定的 `librosa==0.10.2.post1` 无法共存；改为 `modelscope + funasr + kaldiio + addict`，并以 Presence 真实导入/smoke 为验收。

- <https://docs.pytorch.org/get-started/previous-versions/>
- <https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html>
- <https://docs.nvidia.com/cuda/archive/12.4.0/cuda-toolkit-release-notes/>

## 3. 一次性构建

```bash
cd /root/extract-sep

# 新机器缺系统包时，以 root 执行；已安装可省略该变量。
VM_INSTALL_SYSTEM_PACKAGES=1 bash ./setup_env.sh

source ./env.sh
bash ./check_env.sh
```

只有明确需要本机编译 CUDA 扩展时，先安装 Toolkit 12.4，再启用 `nvcc` 硬检查：

```bash
VM_REQUIRE_CUDA_TOOLKIT=1 bash ./setup_env.sh
```

默认环境名是 `ve-cu124`。若同名环境已经存在但 Python 版本不符，脚本会停止，不会自动删除。确认可以重建该环境后再运行：

```bash
VM_RECREATE_ENV=1 VM_INSTALL_SYSTEM_PACKAGES=1 bash ./setup_env.sh
```

也可以用新名字保留旧环境：

```bash
VM_CONDA_ENV=ve-cu124-v2 VM_INSTALL_SYSTEM_PACKAGES=1 bash ./setup_env.sh
```

## 4. DAE-TSE 统一安装

默认 `VM_INSTALL_DAE_TSE=0`，先只构建主流程环境；DAE-TSE 默认禁用，不参与普通分离。明确要做 DAE 实验时再设置 `VM_INSTALL_DAE_TSE=1`，安装脚本会把 `wesep` 与 `kce` 以 editable 模式装入 `ve-cu124`，并写入：

```text
PYTHON_BIN=/root/.../envs/ve-cu124/bin/python
CLEARVOICE_PYTHON=$PYTHON_BIN
DAE_PYTHON_BIN=$PYTHON_BIN
```

已有 DAE 仓若 commit 不符或 tracked files 有修改，脚本不会覆盖它；默认跳过 DAE，硬门模式则停止。可用新路径重装：

```bash
DAE_TSE_REPO=/root/DAE-TSE bash ./setup_env.sh
```

默认情况下，DAE 源码或中文资产异常只会让 DAE 保持 `NO_GO`，不会阻塞 s1–s8 主环境。准备正式验证 DAE 时开启硬门：

```bash
VM_REQUIRE_DAE_TSE=1 \
DAE_TSE_REPO=/实际路径/DAE-TSE \
DAE_TSE_CONFIG=/实际路径/config.yaml \
DAE_TSE_CHECKPOINT=/实际路径/model.pt \
bash ./setup_env.sh
```

官方 `/root/DAE-TSE` 克隆只提供 WeSep/DAE 推理代码；`voice-interaction-challengecup` 也没有实际提交 `tools/build_zh_text_cues.py`，只是引用它。中文运行还需要下面三类外部资产，因此“包可导入”不等于“DAE 可运行”：

```text
$DAE_TSE_CUE_HELPER（参考链通常为 /root/autodl-fs/midea-dae/code/DAE-TSE/tools/build_zh_text_cues.py）
中文 recipe 的 config.yaml（且其 KCE 路径有效）
中文 DAE checkpoint model.pt
```

资产齐全后执行严格 DAE 预检：

```bash
source ./env.sh
python environment/verify_cuda124.py \
  --dae-repo "$DAE_TSE_REPO" \
  --dae-cue-helper "$DAE_TSE_CUE_HELPER" \
  --dae-config /实际路径/config.yaml \
  --dae-checkpoint /实际路径/model.pt \
  --require-dae
```

DAE-TSE 官方发布环境是 PyTorch 2.9.1+cu126；本项目为了满足 CUDA 12.4 和单环境要求，使用 2.6.0+cu124。源码未使用 `torchcodec`，静态接口初查可兼容，但在中文 checkpoint 完成“加载 + 1 条真实音频推理”前只能标记为 **兼容性待验证 / NO_GO**。

## 5. 构建产物

```text
$VM_OUT/meta/env_cuda124_verify.json       # 固定版本、CUDA、ORT、DAE 资产检查
$VM_OUT/meta/requirements-lock-cu124.txt   # 本机完整 pip freeze
./.runtime/python_bin                      # 唯一解释器路径
./.runtime/env.sh                          # 统一环境变量
./env.sh                                   # 上一文件的可 source 副本
```

## 6. Go / No-Go

主流程 `GO` 必须同时满足：

1. Ubuntu、驱动、Python 与固定包版本全部匹配；
2. `torch.version.cuda == 12.4` 且 CUDA 张量计算通过；
3. ONNX Runtime 有 `CUDAExecutionProvider`；
4. `pip check` 无依赖冲突；
5. ClearVoice 与 Qwen-ASR 均从同一 `PYTHON_BIN` 导入。

DAE-TSE 另加：

1. 仓库 commit、中文 helper、配置、KCE 权重和 DAE 权重完整；
2. 同一 `PYTHON_BIN` 可导入 `wesep`、`kce` 并完成中文 cue 转换；
3. checkpoint 严格加载无 missing/unexpected keys；
4. 真实 smoke 输出为 16 kHz、finite、长度与输入一致；
5. 再进入 20 条和全量 CER/声纹安全门评测。

任一 DAE 条件未通过，只禁用 DAE 候选，不得影响已冻结的 s1–s8 主流程。
