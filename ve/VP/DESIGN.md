# VP 实验设计（main 分支历史参考）

> 本文用于解释历史 Voice Presence 实验，不是 `extract@sep` 操作手册。sep 的改进和独立验收边界见 [IMPROVEMENTS.md](../../IMPROVEMENTS.md)。

## 0. 问题边界

**输入：** 干净（相对）KWS enroll `e`，脏 CMD `x`。  
**输出：** `ŷ ∈ {absent, present}`。  
**标签：** pos=`present`，neg=`absent`（仅评估用；推理时不可见）。

不是 TSE，不是 ASR。上一流水线已用 **KWS 文本 oracle CER** 选出 `best_sep`；VP 默认把该文件当 enroll，不在 VP 里重做 cascade。

## 1. 方法骨架（余弦族）

声纹在场检测的标准做法仍是 embedding + 相似度 + 阈值：

```text
e'     = preprocess_enroll(e)          # 因子 A：VAD / 无
{x_k}  = streams(x)                    # 因子 B：mix / 1-sep / 2-sep
s      = max_k cosine(enc(e'), enc(x_k))  # 因子 C：编码器
ŷ      = 1[s ≥ τ]                      # 因子 D：τ 的估计方式
```

**为何仍用余弦：** 短时说话人验证（eres/CAM++/ResNet-LM）的训练目标就是余弦/角度间隔。在你们数据上 AS-Norm、enroll Z-Norm **没有超过 raw max-cosine**。在 raw 被系统打赢之前，不把 PLDA / 质量加权当主线。

**「相似度 → 是/不是」是独立问题：** `s` 的尺度随编码器、是否分离、是否 VAD 而变，**不能**共用一个魔法 τ，也**不能**在全量标定集约最大化代理 contest 当部署 τ（过拟合；且代理把每个 TP 的 CER 当 0，与 VE 真实 accept CER≈0.29 不一致）。

τ 的合法估计：

| 规则 | 做法 | 用途 |
|------|------|------|
| holdout 全局 | 70% 上扫 τ，30% 上报 RR/FRR | **默认** |
| holdout + lang_split | zh/en 各自 holdout 扫 τ | 已有证据更优，须复核 |
| 固定 FAR | 如 FAR=0.15 时的 τ | 稳健工作点，防 contest 过拟合 |
| 全量 max 代理 contest | 禁止当部署 | 只允许当「oracle 上界」对照 |

## 2. 因子与已有证据（先验，不是结论）

| 因子 | 水平 | 先验（旧实验） | VP 中如何处理 |
|------|------|----------------|----------------|
| A enroll VAD | off / energy VAD | VE mix 真实 contest 0.727→0.723，略伤 | **对照因子**，不默认 |
| B CMD 流 | no_sep / sep_once / sep_multi | eres: sep_once ≥ sep_multi > no_sep | 矩阵必跑；sep 只服务打分，不输出给 ASR |
| C 编码器 | eres / camp / resnet | eres ≫ camp ≫ resnet；相关 ~0.85 | 先单模；融合另开一阶 |
| D τ | holdout ± lang_split | lang_split 曾优于全局；抬 τ 真实 contest 下降 | 每 cell 独立 holdout |
| E 融合 | 无 / 灰带 camp | 代理 +0.01；三编码器 OR 伤 RR | **最后做**；须冻结 A–D |

CMD 分离与 KWS cascade **不是同一实验：** KWS 有唤醒文本可对每条轨算 CER；CMD **没有目标文本**，只能 `max cosine` 或日后无文本代理。sep_multi 在 Presence 上已不赚，VP 里只作确认臂。

## 3. 指标（本仓库唯一）

对 neg：`RR = TN/(TN+FP)`，`FAR=1-RR`。  
对 pos：`FRR = FN/(TP+FN)`。  
代理：`proxy = 0.5*RR + 0.5*(1-FRR)`。

报告必须同时给 **calib 子集** 与 **holdout 子集** 的 RR/FRR/proxy。部署看 holdout；calib 上的 proxy 只证明「能拟合」。

样本：pos≈1364，neg≈474。分层抽样若 `LIMIT>0`。

## 4. 实验矩阵（按顺序，一次一因子）

原则：先锁 **eres + sep_once + holdout**，再动 VAD、语言分 τ、换编码器、最后融合。

### 阶段 V0 — 打分基础设施

- 一份 `samples.jsonl`（与 VE 相同契约）
- 若已有 `sep_streams`（`d1/`）：`MODE=matrix ./run_vp.sh`（`score_encoders_on_sep.py`，**不重跑 Moss**；默认 `SKIP_FUSE=1`）
- 否则 `MODE=live ./run_vp.sh`（`calibrate_presence.py` 现打，慢；τ 在 calib 子集选，holdout 上报）
- `summarize_cells.py` 在 **已经选好的 τ** 上再切 holdout 评估。matrix 模式的 τ 仍是全量 in-sample（score_encoders 行为）；**部署 τ 以 live + HOLDOUT_FRAC 为准**

### 阶段 V1 — 主细胞（必跑）

固定：`HOLDOUT_FRAC=0.3`，raw cosine，`select_by=contest` 仅在 calib 上选 τ。

| cell | encoder | streams | enroll_vad | τ |
|------|---------|---------|------------|---|
| V1a | eres | sep_once | off | holdout 全局 |
| V1b | eres | sep_once | off | holdout lang_split |
| V1c | eres | sep_once | on | 与 V1 胜者相同的 τ 模式 |

**门：** 比较 holdout RR/FRR/proxy。VAD 仅当 holdout proxy **不低于** 无 VAD 才进入后续默认。lang_split 同理。

### 阶段 V2 — 流与编码器（在 V1 胜者预处理上）

| cell | 变因 |
|------|------|
| V2-mix | no_sep（仅 mix） |
| V2-multi | sep_multi |
| V2-camp | camp + sep_once |
| V2-res | resnet + sep_once（可 LIMIT 确认） |

**门：** 单模默认仍应是 holdout 上最优的 **eres+流**。camp 即使 proxy 接近 eres，也留到 V3 做灰带，不替换主编码器（相关高，替换增益小）。

### 阶段 V3 — 决策融合（禁止全量再扫 margin）

eres 主判；仅当 `τ-m ≤ s_eres < τ` 时用 camp 救援。`m` 用旧离线 ±0.08 作**唯一**起点，可在 holdout 上微调一档（0.05/0.08/0.10），**禁止**全量网格。  
三编码器 OR/AND：已证明伤 RR，不作候选。

**门：** holdout proxy 相对 V1 胜者 **+0.005** 才保留融合。若日后接入 VE，还须真实 ASR contest 再过一次门（VP 本仓不跑 ASR）。

### 阶段 V4 — 负对照（解释用，不改默认）

- AS-Norm / enroll Z-Norm（旧结论：不优于 raw；复现一格即可）
- 无 holdout 的全量 oracle τ（只报「上界」，禁止部署）

## 5. 运行时决策（部署）

锁定一组 `(encoder, vad, streams, τ_mode, 融合)` 后：

```text
if score < τ(lang): reject
else: accept
```

VE 侧 accept 后默认仍走 **mix ASR**，与 VP 的 streams 解耦：Presence 可以用 sep 轨打分，提取仍用 mix。

## 6. 禁止

- 用 VP proxy 当 VE 竞赛分
- 全量集 max contest 当 τ
- 未做 V1 就上融合或换 TSE
- 把 KWS 文本 CER 选轨搬到 CMD
- 默认打开 VAD / sep_multi / 三编码器投票
