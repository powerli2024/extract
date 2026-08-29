# 改进方案与验收路线

## 1. 当前判断

现有实现已经具备完整时长分离、批量 OOM 降级、阶段内/跨阶段 CER 选轨、门控 cohort 去重、断点续跑和下游 handoff。主要风险不再是“能不能跑”，而是四件事：

1. 续跑结果缺少完整实验签名，配置变化可能误复用旧结果；
2. `best_sep` 由同一 Qwen oracle CER 选出，容易把开发 ASR 偏好当成真实纯净度；
3. 全量 1,838 是脚本外部硬编码，数据版本变化时可能出现错误验收；
4. s1–s8 全局 oracle 是候选上界，尚未形成 s1 优先、条件升级、可生产复算的冻结路由。

因此改进顺序应是先补证据与契约，再优化选路，最后才考虑增加处理后端。当前不引入 Conv-TasNet。

## 2. P0：文档与契约冻结（已可立即执行）

目标：让任何人都能回答“这轮用了什么数据、模型、代码、指标，为什么可续跑”。

改进项：

- 以根 [README.md](README.md) 为唯一入口，`ve/` 文档明确标成 `main` 历史参考；
- 固定中文无调拼音 CER、英文字符 CER，并在每条 index 保存 metric；
- handoff 固定 schema v2、全波形、不截断、20 ms 时长容差；
- 冒烟、覆盖审计、效果验证三类结论分开写。

**Go：** 新人只按 [OPERATIONS.md](OPERATIONS.md) 能完成 smoke，并能准确找到 index、WAV、审计与 handoff。  
**No-Go：** 仍需从 `ve/` 文档猜入口，或把开发 oracle 写成生产分数。

## 3. P1：运行签名与防污染续跑（最高优先级代码改进）

为每轮根目录写 `run_signature.json`，至少包含：

```text
data: pos/neg JSONL sha256、UID 数、源 WAV 清单 hash
code: git commit、dirty diff hash
models: ONNX/SS/Qwen 文件或目录 fingerprint
config: stages、peak、sample rate、ASR 参数、CER 口径、全波形策略
runtime: Python、torch、onnxruntime、CUDA provider
```

每个阶段 summary 保存该签名 hash。续跑前比较签名：一致才复用；不一致则停止并要求新 `VM_OUT`。不要自动覆盖旧证据。

**Go：** 人为改变任一模型、CER 口径或数据 JSONL 后，旧目录续跑必然在分离前失败；同签名中断可继续。  
**No-Go：** 只比较 UID 数，或用 `--force` 自动吞掉不一致。

## 4. P2：覆盖数从 manifest 派生

把 `expected_uids=1838` 从主判定改为当前 collect manifest 的总数，同时保留 1,364/474 作为已知 DatasetA 版本的显式基准。

建议同时输出：

- `dataset_signature.json`；
- pos/neg 各自 expected、observed、missing、extra、duplicate；
- source WAV hash 与时长摘要。

**Go：** 数据增删一条时审计能指出具体 UID，且不会把旧的 1,838 当作永恒常量。  
**No-Go：** 只把命令行数字改成另一个硬编码数字。

## 5. P3：音频与 ASR 去重

同一 UID 的 original/分离轨可能字节完全相同。ASR 缓存键建议固定为：

```text
(audio_sha256, wake_text, lang, asr_signature)
```

先按规范 PCM hash 去重，再只识别一次；所有候选引用同一结果。若旧结果中同 hash 出现不同 finite CER，应写入冲突报告并禁止直接排行。

收益主要是：减少计算、消除随机 ASR 重复评分冲突、让跨阶段 paired delta 可解释。

**Go：** 相同 hash 的候选只产生一个 ASR 记录且 CER 完全一致；覆盖仍是 100%。  
**No-Go：** 用路径作缓存键，或只在同一阶段内去重。

## 6. P4：冻结 s1 优先的可生产选路

`best_sep` 的全局 oracle 可保留为开发上界，但生产候选应另建一条可独立复算的路由：

```text
s1 baseline
  └─ 仅满足冻结困难条件时比较 s7
       └─ s7 严格改善且安全门通过才切换
            └─ 可选 SE 只作为第三候选视图
```

重要实现事实：当前 s7 的计算父节点是 s2，所以“选择上 s1→s7”并不等于“计算上跳过 s2”；实际依赖仍是 `s2→s7`。文档和报告必须同时写清计算图与选择优先级。

建议冻结规则：

- s1 CER 非零或进入预先定义困难桶，才允许比较 s7；
- s7 只有严格改善才替换，平局保留 s1；
- 不用 `cos(candidate, raw)` 作为“纯净度”主排序，它只可作为灾变回退门；
- 若增加 SE，必须同时通过 raw-speaker cosine 与 post-SE CER 安全门；逐条 SE 失败时回退冻结候选并记录原因。

**Go：** 对全部 UID 单独运行路由复算，覆盖 100%，选择原因可追溯；冻结 KWS/Presence/CMD 指标稳定提升。  
**No-Go：** 只展示被改善的子集、允许平局切换，或把全局 oracle winner 当运行时可得决策。

## 7. P5：独立效果验收

开发期至少报告三层证据：

| 层级 | 要回答的问题 | 能否批准生产 |
|---|---|---|
| extract-sep oracle | 候选池是否存在更低唤醒词 CER | 否 |
| KWS/声纹安全门 | 候选是否保留目标说话人、是否优于 s1 | 仍不足 |
| 冻结 Presence/CMD/contest | 最终系统是否真实提升且无覆盖回归 | 是 |

正式 VE 评测应在 `/root/extract@main` 使用严格设置：`STRICT_ENROLL=1 STRICT_EVAL=1 ASR_RESUME=0 LIMIT=0`，并由 `final_evaluate.py --strict` 以字符加权 micro CER 给出最终结论。历史 overlay、ASR resume 和不同输出目录不得混入。

建议批准门：

- 全量覆盖与 finite-CER 覆盖均与基线相等；
- 正式 contest 不低于冻结基线，并达到预先约定的最小增益；
- 语言、噪声、重叠和困难桶无不可接受回归；
- 结果在新输出目录独立复算，不依赖手工挑选；
- 任一硬门失败即写 `NO_GO`，不以平均提升覆盖缺失或退化。

## 8. 建议实施顺序

| 顺序 | 工作量 | 价值 | 建议 |
|---|---:|---:|---|
| P1 运行签名 | 中 | 很高 | 先做 |
| P2 manifest 派生覆盖 | 小 | 高 | 与 P1 同批 |
| P3 hash 去重 ASR | 中 | 高 | 第二批 |
| P4 s1 优先冻结路由 | 中 | 很高 | 先设计后全量复算 |
| P5 独立系统验收 | 大 | 决定性 | 每个候选必跑 |

性能优化（降低窗口/批次密度、压缩报告）排在效果与稳定性之后。当前阶段优先保证结果可复现、覆盖完整和结论不越界。

## 9. DAE-TSE 当前落地状态

已增加隔离候选入口 [DAE_TSE.md](DAE_TSE.md)：支持 raw/s1–s8 source、完整输入签名、统一 cu124 Python、精确同长度审计和 source/DAE 同轮 CER。它仍未实现声纹安全门，也没有 AutoDL 全量模型结果，因此状态是代码候选而非效果通过。

下一步先跑 DAE(s1) 全量 paired 评估；只有严格改善比例、CER=0 保持和坏例回归可接受，才加入 raw-speaker cosine 安全门，再与冻结 s1→s7 路由组合。MMS-FA 裁切、DAE、SE48K 和 V7 gate 应分臂逐项验证，不一次性整体搬入。
