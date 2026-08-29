# AutoDL 运行与验收手册

本手册只适用于 `extract@sep`。技术原理见 [TECHNICAL_FLOW.md](TECHNICAL_FLOW.md)。

## 1. 推荐目录

```text
/root/extract                 # main：Presence / CMD / 比赛评测
/root/extract-sep             # sep：本手册的 KWS 分离
/root/kws                     # 下游独立选路与审计（推荐同步）
/root/miniconda3/envs/ve      # 三者可共用 Python，但代码与输出分开

/root/autodl-tmp/
  datasetA/
  checkpoints/MossFormer2_ONNX/simple_model.onnx
  checkpoints/MossFormer2_SS_16K/last_best_checkpoint.pt
  Qwen3-ASR-1.7B/
  cache/{huggingface,torch,pip}/
  kws_sep_<run_id>/            # 每轮独立 VM_OUT
```

不要让 `/root/extract` 与 `/root/extract-sep` 共用工作目录，也不要在任一克隆内来回 checkout 分支。

## 2. 首次安装

```bash
cd /root
git clone -b sep https://github.com/powerli2024/extract.git extract-sep
cd /root/extract-sep
chmod +x ./*.sh

export DATA_DIR=/root/autodl-tmp/datasetA
export VM_OUT=/root/autodl-tmp/kws_sep_20260829
export MOSS_CKPT_DIR=/root/autodl-tmp/checkpoints
export ASR_MODEL_DIR=/root/autodl-tmp/Qwen3-ASR-1.7B

bash ./setup_env.sh
source ./env.sh
./download_models.sh
bash ./check_env.sh
```

`check_env.sh` 返回 0 才表示全部阶段就绪；返回 2 代表只能跑部分阶段，不能开始正式全量。

## 3. 冒烟与全量

冒烟用于发现依赖、显存、路径和格式问题：

```bash
bash ./rerun_and_audit.sh --smoke
```

正式评测必须新建或确认独立的 `VM_OUT`，再跑：

```bash
bash ./rerun_and_audit.sh --full
```

若希望先冒烟、再在同一新目录续到全量：

```bash
bash ./rerun_and_audit.sh --all
```

运行中断后，使用相同命令会保留当前目录中已成功的 UID 并补齐其余项。模型、代码、数据、ASR 配置或 CER 口径发生变化时，不要继续旧目录，应改用新的 `VM_OUT`。

## 4. 分阶段排障

```bash
./run_stage.sh collect
./run_stage.sh s1 --limit 20
./run_stage.sh s2 --limit 20
./run_stage.sh s3 --limit 20
./run_stage.sh s4 --limit 20
./run_stage.sh s5 --limit 20
./run_stage.sh s6 --limit 20
./run_stage.sh s7 --limit 20
./run_stage.sh s8 --limit 20
./run_stage.sh compare
./run_stage.sh eval
./run_stage.sh handoff
```

只补失败项：

```bash
./run_stage.sh s2 --retry-failed
```

`--force` 会重算目标阶段，可能覆盖证据，只应在确认使用新的输出目录或确实需要整阶段重算时使用。

## 5. 严格验收

本仓审计：

```bash
python scripts/audit_sep_run.py \
  --vm-out "$VM_OUT" \
  --expected-uids 1838
```

若已部署 `/root/kws`，继续验证交接契约：

```bash
python /root/kws/scripts/audit_sep_input.py \
  --pos-neg "$VM_OUT" \
  --expected-uids 1838 \
  --check-duration \
  --require-handoff \
  --out "$VM_OUT/reports/kws_input_audit.json"
```

验收表：

| 检查 | PASS | FAIL 后动作 |
|---|---|---|
| 数据覆盖 | 当前 DatasetA 全 UID；已知版本为 1,838 | 停止 handoff，查 collect/数据版本 |
| s1–s4 | 每 UID 有成功 index 与所需 WAV | `--retry-failed`，不可静默跳过 |
| s5–s8 | UID 数与父阶段门控计划一致 | 查阈值、duplicate_of、父 index |
| metric | 中文 pinyin、英文 char | 换新 `VM_OUT` 全量重算 ASR |
| 时长 | 输出与源差 ≤20 ms | 停止采用，查模型输出对齐 |
| best_sep | index/WAV/handoff 一致，n_fail=0 | 停止下游使用 |
| 独立效果 | 冻结 KWS/Presence/contest 达标 | 否则明确 `NO_GO` |

## 6. 下游交接

审计通过后，`/root/kws` 读取整个 `$VM_OUT`，而不是只读散落的 WAV：

```bash
cd /root/kws
python scripts/audit_sep_input.py \
  --pos-neg "$VM_OUT" --expected-uids 1838 \
  --check-duration --require-handoff
python scripts/rebuild_best_sep.py --pos-neg "$VM_OUT"
python scripts/analyze_dual_zero.py --pos-neg "$VM_OUT"
```

若要给 `extract@main` 提供 enroll，应使用下游独立验证通过的目录，再建立明确软链；不要默认把 `extract-sep/best_sep` 直接提升为生产目录。

## 7. 常见问题

| 症状 | 含义与处理 |
|---|---|
| `check_env.sh` 返回 2 | 缺某个后端或权重，只能局部冒烟，不能正式全量 |
| `CUDAExecutionProvider` 缺失 | 当前 Python 的 ORT 不是固定 GPU 版；重建统一 `ve-cu124` 环境 |
| `clearvoice` import 失败 | s2/s4/s5/s8 不可用；安装 optional requirements 后重检 |
| 批量 OOM | 程序会自动二分；仍失败的单条必须进入 error，不能裁短 |
| 门控 a/b/c 数量相同 | 拼音 CER 离散导致同 cohort，`duplicate_of` 是预期行为 |
| 续跑结果看似异常 | 先检查输出目录和配置是否混用；不要用 `--force` 掩盖签名冲突 |
| `./ve.sh` 退出 | 正常；Presence/比赛评测应在 `/root/extract@main` |

## 8. 交付清单

一次可复现交付至少包含：

- 代码 commit 与 `git status`；
- 本轮环境变量和模型位置；
- `$VM_OUT/kws_handoff.json`；
- `$VM_OUT/best_sep/index.jsonl` 与 WAV；
- `audit_sep_run.json`、`kws_input_audit.json`；
- 全阶段比较与同 UID 冲突报告；
- 最终独立验证结论（`GO` 或 `NO_GO`）。

## 9. 可选 DAE-TSE

DAE-TSE 与主流程共用 `ve-cu124` Python，但仍是隔离的可选候选，不属于 `./run_sep.sh` 的资产硬依赖。s1–s8 全量审计通过后，再按 [DAE_TSE.md](DAE_TSE.md) 使用新的 `DAE_OUT` 运行；不得把 smoke 目录续成正式结果，也不得手工把实验 WAV 塞进 `best_sep`。
