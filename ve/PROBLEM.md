# VE 问题定义与分阶决策（main 分支历史参考）

> **分支边界：** `extract@sep` 不执行 VE/Presence/比赛评测。当前 KWS 分离流程见 [仓库根 README](../README.md)；本文中的历史数值不能直接作为当前生产证据。

本文是项目的**问题导向**文档：先固定要解什么、用什么指标、已有证据说了什么，再规定每一阶实验的 Go/No-Go。实现细节见 `README.md` / `AUTODL.md`；端到端怎么跑见 [`EXPERIMENTS.md`](EXPERIMENTS.md)。

**拒识（说话人是否在 CMD）已拆到独立模块 [`VP/`](VP/README.md)**（Voice Presence）。VP 只优化检测；过门后再进 VE 的 mix ASR。VP 实验设计见 [`VP/DESIGN.md`](VP/DESIGN.md)。

**默认提交臂：Presence + mix ASR。** 未过门的方向不得改默认。

---

## 1. 问题与指标

### 1.1 任务

运行时**不知道**样本是正还是负。给定：

| 输入 | 含义 |
|------|------|
| KWS enroll | `pos_neg/best_sep/{split}/{uid}.wav`：上一流水线按唤醒文本 oracle CER 选出的较干净注册声 |
| 注册文本 | 唤醒词 / 语言（用于 ASR 语种与 CER 的 ref 侧 cmd 文本来自 datasetA） |
| CMD | `datasetA/{pos,neg}/cmd_*.wav`：可能含他人、噪声、重叠的脏混合音 |

输出：拒识，或一段波形再 ASR。拒识口径**仅**「enroll 对应说话人不在 CMD 中」。

```text
enroll + cmd → Presence
                 ├─ absent → REJECT（正样本 CER=1）
                 └─ present → 波形（默认 mix）→ Qwen3-ASR → CER
```

### 1.2 竞赛指标（唯一口径）

- **负样本**：`RR = n_reject / n_neg`（只统计拒识率）
- **正样本**逐条：
  - 误拒 → `CER_i = 1`
  - 接受且有识别文本 → `CER_i = (S+D+I) / N`（与 VM 一致：NFKC、去空白标点、小写；取值 0–1）
- **总分**：`contest = 0.5 * RR + 0.5 * (1 - mean_i CER_i)`

正样本平均 CER 记为 `CER_total`。接受子集上的平均记为 `CER_accept`（不含误拒的 1）。

### 1.3 分解（用来选优化方向，不是第二套指标）

对正样本：

```text
CER_total ≈ FRR * 1 + (1 - FRR) * CER_accept
contest   = 0.5 * RR + 0.5 * (1 - CER_total)
```

因此：

- 压 FRR：减少「CER=1」的误拒份额，但可能降 RR（FAR 升），且新放行样本的真实 CER 未必接近 0
- 压 CER_accept：只改善已接受正样本；**盲 TSE 若抬高 CER_accept，竞赛分必掉**
- **Presence 代理分**把每个 TP 的 CER 当成 0、每个 FN 当成 1，与真实 `CER_accept ≈ 0.29` 不一致 → **禁止用代理 contest 指导提交**

### 1.4 两套数字不要混用

| 名字 | 何时出现 | 能否当提交依据 |
|------|----------|----------------|
| Presence 代理 contest | 校准 / encoder 扫 thr，CER:=FRR | 否 |
| 真实 contest | `reports/asr_cer/`，Qwen3-ASR | **是** |

---

## 2. 已有证据

数据源：四 PIPELINE 全量 ASR（共享 Presence：eres2netv2 + sep_once + lang_split，当时无 enroll VAD 或 VAD 未进默认）见仓库外 `qqqqqqqqqqqqqq/ve_all_results/`；encoder 对照见 `ve_encoder_cmp_reports/`。

### 2.1 端到端（同一门控，只换 accept 后波形）

| 臂 | contest | CER_total | CER_accept | RR | FRR |
|----|---------|-----------|------------|------|------|
| **mix** | **0.727** | 0.412 | 0.289 | 0.865 | 0.172 |
| sep_route | 0.716 | 0.433 | 0.314 | 同门控 | ~0.173 |
| wesep | 0.702 | 0.461 | 0.347 | 同 | ~0.174 |
| ps4 | 0.671 | 0.523 | 0.423 | 同 | ~0.173 |

- 门控几乎相同（决策一致率 ≥99.8%）；**差距全在提取质量**
- mix：CER_total ≈ 误拒份额 0.172 + accept 贡献 ~0.240
- 中文损伤大于英文（ps4 zh accept CER 0.34→0.50）
- 逐条 oracle 选最低 CER 后端：contest 上界 **0.765（+0.038）**；最优次数 mix 948 / sep_route 77 / wesep 75 / ps4 29

### 2.2 Presence / 编码器（代理分，n=1838）

单模（own thr）：**eres + sep_once 0.837** > sep_multi 0.834 > no_sep 0.830；camp / resnet 更低。  
eres–camp 分数相关 ~0.85；camp 可救 ~81 条 eres FN。  
灰带 eres∨camp（±0.08）代理 contest ~**0.849**；三编码器 OR 伤 RR（~0.74），代理分反而差。  
AS-Norm / enroll Z-Norm **未超过 raw cosine**。

### 2.3 Enroll VAD

能量 VAD 只裁 enroll。mix 真实 contest **约 0.727 → 0.723**（略降）。不得因「看起来干净」默认开启而不对照。

### 2.4 抬 thr

仅抬 Presence thr + 真实 ASR：最佳约 **0.717 < 0.727**。与代理「再抠 thr」方向相反。

### 2.5 粗算上限（量级，非承诺）

在 mix 的 CER_accept≈0.29、RR≈0.865、FRR≈0.17 附近：抽干误拒或抽干 accept CER，contest 都可能到 **~0.84–0.85**。当前 0.727。盲 TSE 相对 mix 为负；oracle 路由只 +0.04。**主缺口在门控误拒 + mix 上难例 ASR，不在再下一套全量 TSE。**

---

## 3. 对现有思路的批评

### 3.1 「KWS 再加 VAD / 再加工序」

`best_sep` 已按 **KWS 文本 oracle CER** 选轨。再在 VE 里加能量 VAD，是在已选净的波形上切静音；实测真实 contest 略降。

应先问 **enroll 是否已饱和**（VAD 实际裁掉时长、失败回退率、与 raw kws 的 sim、差 enroll 是否对齐高 FRR uid）。不饱和 → 改 **上一流水线（onnx/cv cascade）**，不要把不确定增益堆进 VE 默认路径。

### 3.2 「声纹提升空间大」——半对，优化对象错了

空间确实大：FRR≈17%；|score−thr|<0.05 约 108 条。  
错在把 **全量集 contest-optimal thr** 当可泛化超参，以及用 **代理 contest** 当验收。

余弦 + 单编码器对短唤醒仍是合理基线（eres 面向中文短时 SV）。AS-Norm 等未赢 raw。下一步：

- thr：**holdout** 或固定 FAR，禁止全量 max contest
- 编码器：**eres 主判**；camp **仅灰带救援**；禁止三编码器 OR 作默认
- 融合必须以 **真实 ASR contest** 过门（§4 P1）

### 3.3 「KWS 多次 sep 搬到 CMD」

KWS cascade 有效是因为 **有唤醒文本 → 可对每条轨算 CER**。CMD **运行时无目标文本**，不能复制 oracle 选轨。盲 sep_route 已全量 **-0.011 contest vs mix**。

CMD 多次 sep 只允许 **难例子集 + 无文本代理选路**（enroll sim / ASR 置信度），默认回退 mix。

### 3.4 「换更强 TSE」当主线

PS4/WeSep 全面伤 ASR。更可能：域差（英文混合预训练 vs 中英短指令）、畸变对 CER 比 SI-SDR 更敏感、Qwen3 对 mix 已较鲁棒。USEF 另有 8 kHz 与 NC 许可问题（[`NOTES_USEF_RESAMPLE.md`](NOTES_USEF_RESAMPLE.md)），不得替换默认臂。

### 3.5 项目曾为流水线服务

`run_all` 把门控、提取、ASR 绑死，用代理分驱动提交。原则改为：**一次实验只动一个因子；默认提交永远是 Presence + mix。**

---

## 4. 分阶实验与 Go/No-Go

只跑过门的才进入下一阶。命令见 [`EXPERIMENTS.md`](EXPERIMENTS.md)。ASR 换模型本轮搁置（P5）。

### P0 冻结口径与基线

- 固定 §1 指标与 CER 归一化
- 基线：eres + sep_once Presence + **mix** ASR；`HOLDOUT_FRAC=0.3` 选 thr
- 对照：`ENROLL_VAD=0` 同其余配置

**Go：** 两套都有真实 `asr_cer/summary`；记下 contest / RR / FRR / CER_total。VAD 仅当真实 contest **不低于** novad 才进入默认。

### P1 拒识（唯一高优先级算法题）

**同一 mix 波形**，只换门控：

1. holdout 全局 thr vs lang_split
2. eres vs camp vs 灰带 OR（margin 以离线 ±0.08 为起点，**禁止**全量再穷举）
3. 报告 RR、FRR、**真实** CER_total、contest；灰带多放行的 FP 是否拖垮 contest

**Go：** 真实 contest **稳定 ≥ P0 mix 基线 +0.005** 才改默认门控。否则保持 P0。

### P2 KWS 纯净度（诊断，默认不加工序）

抽查 best_sep：裁切比例、回退、与 FRR uid 对齐。  
**Go（回 KWS 仓改 cascade）：** 差 enroll 与误拒显著对齐。  
**No-Go：** 几乎不裁或与 FRR 无关 → VE 不再叠 VAD。

### P3 CMD 多次 sep（仅难例）

难例：mix accept 且 CER>0.5，或 Presence 灰带。对这些 uid 跑 d1/d2，**enroll sim 选轨**，与 mix 比 CER。全量不跑。  
**No-Go：** 难例上仍多数 mix 更好 → 关闭该方向。

### P4 TSE（仅难例 LIMIT）

mix vs sep_route vs（可选）USEF 冒烟。主提交不换。  
**Go 接入默认：** 难例真实 CER 稳定优于 mix，且全量 contest 不下降。当前证据下预期 No-Go。

### P5 ASR

换识别模型：搁置，不排期。

---

## 5. 默认提交与禁止事项

**默认：** `PRESENCE_BACKEND=eres2netv2`，`USE_SEP=1`（sep_once），`LANG_SPLIT=1`，score_norm=raw，`PIPELINE=mix`，Qwen3-ASR-1.7B。Enroll VAD 以 P0 对照为准。

**禁止（未过门前）：**

- 全量默认 ps4 / wesep / sep_route / USEF
- 在全量标定集约最大化代理 contest 当部署 thr
- 三编码器 AND/OR 作默认门控
- 把 Presence 代理分写成「竞赛成绩」
- 用 KWS 的文本 oracle CER 在 CMD 上选轨（运行时没有该文本）
