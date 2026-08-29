# 技术处理流程

本文解释 `./run_sep.sh` 从 DatasetA 到 `best_sep` 的真实代码路径。部署命令见 [OPERATIONS.md](OPERATIONS.md)，跨仓字段见 [KWS_HANDOFF.md](KWS_HANDOFF.md)。

## 1. 输入与身份约束

输入由两部分组成：

```text
$DATA_DIR/
  pos.jsonl、neg.jsonl
  pos/kws_<id>.wav
  neg/kws_<id>.wav
```

`collect_kws.py` 以 JSONL 中的 `id` 生成稳定 UID：`{split}_{id}`，例如 `pos_0`。它不允许用行号替代 id，也不允许重复 id、跨 split UID 或静默路径回退。这样可以保证后续每个 index、WAV 和报告都能回到同一条原始数据。

正式数据还应满足：

- `pos.jsonl`、`neg.jsonl` 的 `id` 唯一；
- `唤醒音频` 指向的文件与 `kws_<id>.wav` 一致；
- 每条有可用唤醒文本；
- 输入音频完整可读。

## 2. 完整时长分离

所有分离阶段都读取完整 KWS 波形。`--max-sep-sec` 仅为旧命令兼容参数，不再裁切输入。

ONNX 默认按时长排序后组成批次，以减少长短音频混批造成的显存浪费。批量 OOM 时自动二分降批；单条 OOM 会清理显存并以完整波形再试一次。仍失败则在 index 中记录 `error`，不会拿截断音频继续评分。

分离输出只允许对模型输出做尾部 trim/pad，使其与输入采样点数一致。严格审计要求输出与源音频时长差不超过 20 ms。

## 3. s1–s4：全量候选

### s1 / s2：一阶分离

两条臂输入相同，后端不同：

```text
original KWS ──► s1 ONNX ─────────► original + spk1 + spk2
             └─► s2 ClearVoice ───► original + spk1 + spk2
```

`original` 在磁盘上的 tag 为 `peak`，表示按峰值 0.7 归一化后的原始波形。保留 original 很重要：分离并不必然改善短唤醒词，原波是无损回退臂。

### s3 / s4：级联分离

s3 读取 s1 的 spk1/spk2，再分别做一次 ONNX 分离；s4 对 s2 做同样的 ClearVoice 级联：

```text
parent spk1 ──► spk1_r1 + spk1_r2
parent spk2 ──► spk2_r1 + spk2_r2
```

每条 UID 的候选池包含：original、父阶段 spk1/spk2、二阶四轨，共七条。父阶段 WAV 缺失时直接失败，禁止回退到另一份原始音频掩盖上游缺口。

## 4. ASR 与 CER

每条候选都送入同一 Qwen3-ASR 配置，再按唤醒文本计算 CER：

| 语言 | 排序指标 | 目的 |
|---|---|---|
| 中文 | 无调拼音 CER (`cer_py`) | 降低同音字对短唤醒词选轨的干扰 |
| 英文 | 规范化字符 CER (`cer_char`) | 与当前 KWS 开发口径一致 |

这不是 `/root/extract@main` 的比赛严格中文字符 CER。两套口径服务不同目标，不能混写成绩。

阶段内选择规则为：

```text
oracle_stream = argmin(candidate CER)
若 CER 相同：保留 original
```

因此阶段 index 既保存所有流的 hyp/CER，也保存 `oracle_stream`、`oracle_cer` 与实际 metric。这里的 “oracle” 明确表示用到了真值唤醒文本，只适合离线开发与生成候选。

## 5. s5–s8：高 CER 子集抢救

s5–s8 不对所有样本重复计算。它们先读取父阶段的 `oracle_cer` 分布，默认使用 P50、P75、P90 形成 a/b/c 三个阈值，只处理 `parent_oracle_cer >= threshold` 的 UID。

| 阶段 | 父阶段 | 动作 |
|---|---|---|
| s5 | s1 | 用 ClearVoice 重新分离父阶段 original |
| s6 | s1 | 用 ONNX 继续分离父阶段 spk1/spk2 |
| s7 | s2 | 用 ONNX 继续分离父阶段 spk1/spk2 |
| s8 | s2 | 用 ClearVoice 继续分离父阶段 spk1/spk2 |

短唤醒词的拼音 CER 取值离散，P50/P75/P90 可能选出完全相同的 UID 集合。`gate_policy.py` 对 UID cohort 做签名；相同 cohort 只计算第一个 a/b/c 臂，其余在 summary 中写 `duplicate_of`。这既避免重复分离，也避免同一 PCM 被随机 ASR 多次打出冲突结果。

注意：a/b/c 是计算预算门控，不是生产 Presence 阈值，也不是经独立验证的部署策略。

## 6. 比较、报告与跨阶段导出

`compare` 和 `eval` 汇总：

- s1–s4 的全量 mean CER、分布和两两差异；
- s5–s8 在同一门控 UID 上相对父阶段的 paired delta；
- 错误数、缺失数和每个阈值 cohort 的覆盖。

`handoff` 会发现各阶段及非重复的 `thr_*` index，对同一 UID 收集候选，再执行：

```text
1. oracle_cer 更小者优先
2. CER 平局时，非 original 优先
3. 仍平局时，按稳定的阶段发现顺序优先
```

最终波形复制到 `$VM_OUT/best_sep/{pos,neg}/{uid}.wav`，并写：

- `best_sep/index.jsonl`：只含 `ok=true` 的可用记录；
- `best_sep/failed.jsonl`：缺失源 WAV 等失败；
- `kws_handoff.json`：schema、分支、全波形策略、阶段目录和记录数；
- `best_sep/kws_handoff.json`：便于只搬 best_sep 时仍携带契约。

跨阶段平局优先分离轨只是开发期 tie-break，不代表分离轨一定更纯净。独立验证若不赢，应保留 s1/original 基线。

## 7. 续跑与失败语义

默认 `VM_SKIP_DONE=1`。同一输出树内，已成功 UID 会保留，缺失或失败 UID 可用 `--retry-failed` 补齐；`--force` 才会强制重算。ClearVoice 阶段先完成分离并释放模型，再加载 ASR，降低同卡 OOM 风险。

当前续跑机制主要检查 index 覆盖与父阶段计划，尚未完整绑定代码 commit、权重 hash、ASR 参数和数据 hash。因此：

- 配置或模型变化时必须换新的 `VM_OUT`；
- 不要从旧实验手工复制 index/WAV；
- 新旧结果只能通过显式比较脚本关联。

完整的签名改进见 [IMPROVEMENTS.md](IMPROVEMENTS.md)。

## 8. 严格审计链

`rerun_and_audit.sh --full` 的验收顺序是：

```text
test_sep_invariants.py
  → check_env.sh
  → run_sep.sh（同树续跑到全量）
  → audit_sep_run.py（覆盖、metric、WAV、时长、门控 cohort）
  → kws/audit_sep_input.py（若 /root/kws 可用）
  → 全阶段比较、同 UID 音频排行、旧新 best_sep 对照
```

审计通过只证明“这次计算完整、契约一致”，不证明模型效果已经提升。效果结论必须来自冻结的独立 KWS/Presence/CMD/比赛评测。

## 9. 系统边界

三个仓/克隆的职责必须分开：

| 位置 | 职责 | 不能做什么 |
|---|---|---|
| `/root/extract-sep` (`sep`) | KWS 分离、开发 CER、候选交接 | 不跑 Presence/比赛提交 |
| `/root/kws` | s1 优先的下游选路、声纹/CER 安全门、独立比较 | 不把 oracle best 直接当生产 best |
| `/root/extract` (`main`) | Presence、CMD、ASR、严格比赛评测 | 不在该克隆切换到 `sep` |

这条边界用于避免历史 overlay、ASR resume 或不同分支产物相互污染。

## 10. 可选 DAE-TSE 分支

`run_dae_tse.sh` 可读取 raw、s1–s4 全量 winner，或 s5–s8 的指定 `thr_a/b/c` cohort，将冻结 source 与已知唤醒文本交给中文 DAE-TSE。DAE 与常规流程共用固定 `ve-cu124` Python，模型在一次任务中只加载一次；输出必须保持 16 kHz 和精确采样点数，再对 source/DAE 同轮重算 CER。

该分支写入 `$VM_OUT/experiments/`，不参与 `export_kws_handoff.py` 的阶段发现。技术细节与命令见 [DAE_TSE.md](DAE_TSE.md)。
