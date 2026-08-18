# VE 分阶实验（AutoDL，现有开关）

对照 [`PROBLEM.md`](PROBLEM.md)。**不改脚本**；只列环境变量。一次只动一个因子。

前置：代码在 extract 仓的 `ve/`（AutoDL：`/root/extract/ve`）；数据与权重仍在 `/root/autodl-tmp/`。

```bash
cd /root/extract/ve   # extract.git 内嵌的 VE
source .env_ve
# 看全部变量: ./run_all.sh --help
```

验收数字只认 `$VE_OUT/reports/asr_cer/summary.md` 里的 **contest / RR / CER_total**，不认校准 JSON 里的代理 contest。

---

## P0 基线：Presence + mix

### P0a — VAD 开 + holdout 选 thr（推荐先跑）

```bash
FORCE_CALIB=1 HOLDOUT_FRAC=0.3 ENROLL_VAD=1 PIPELINE=mix ./run_all.sh
```

产物：

- 提取 / ASR：`/root/autodl-tmp/ve_mix_vad/`
- 校准：`/root/autodl-tmp/ve_presence_best/reports/presence_calib_eres2netv2_sep1_ls_vad_raw/`
- 看校准里的 `holdout.contest_score`（代理，仅参考）+ `ve_mix_vad/reports/asr_cer/summary.md`（真实）

默认已含：`PRESENCE_BACKEND=eres2netv2`、`USE_SEP=1`、`LANG_SPLIT=1`、raw 分数。

### P0b — 无 VAD 对照（只改 ENROLL_VAD）

```bash
FORCE_CALIB=1 HOLDOUT_FRAC=0.3 ENROLL_VAD=0 PIPELINE=mix ./run_all.sh
```

产物：`/root/autodl-tmp/ve_mix_novad/` 与 `…_ls_novad_raw/` 校准桶。与 P0a **并存**。

**P0 门：** 两套都有真实 ASR。VAD 仅当 `ve_mix_vad` 真实 contest **≥** `ve_mix_novad` 才作为后续默认；否则后续实验用 `ENROLL_VAD=0`。

冒烟（可选）：`LIMIT=64 SKIP_ASR=1` 只检查流程；正式对比 `LIMIT=0` 且跑 ASR。

---

## P1 拒识（同一 mix，只换门控）

P0 门控选定后，**固定** `ENROLL_VAD` 与 `PIPELINE=mix`。换 thr 模式或编码器时必须 `FORCE_CALIB=1`（换校准桶）。

### P1a — lang_split vs 全局 thr

```bash
# 全局单 thr（对照）
FORCE_CALIB=1 HOLDOUT_FRAC=0.3 LANG_SPLIT=0 ENROLL_VAD=<P0胜者> PIPELINE=mix ./run_all.sh
```

`LANG_SPLIT=0` 时校准目录标签为 `gthr` 而非 `ls`，与 P0 的 `ls` 桶不覆盖。  
**勿**再设 `VE_OUT` 为同一路径：`run_all` 默认 `ve_mix_vad` 会覆盖 extract。若 P0a 已占用 `ve_mix_vad`，对照请显式分开：

```bash
VE_OUT=/root/autodl-tmp/ve_mix_gthr_vad \
CALIB_DIR=/root/autodl-tmp/ve_presence_best/reports/presence_calib_eres2netv2_sep1_gthr_vad_raw \
FORCE_CALIB=1 HOLDOUT_FRAC=0.3 LANG_SPLIT=0 ENROLL_VAD=1 PIPELINE=mix ./run_all.sh
```

（`ENROLL_VAD=0` 时把路径里的 `vad` 改成 `novad`。）

### P1b — 换 Presence 编码器（仍 mix ASR）

```bash
VE_OUT=/root/autodl-tmp/ve_mix_campplus_vad \
PRESENCE_BACKEND=campplus \
FORCE_CALIB=1 HOLDOUT_FRAC=0.3 ENROLL_VAD=1 PIPELINE=mix ./run_all.sh
```

校准会落到 `presence_calib_campplus_sep1_ls_vad_raw/`。与 eres 比的是**真实 contest**，不是校准代理分。

resnet 同理：`PRESENCE_BACKEND=resnet34_lm` + 独立 `VE_OUT`。预期弱于 eres，只作确认不必全量。

### P1c — 灰带 eres∨camp

**现有 `run_all.sh` 没有灰带融合开关。** 本阶不要在全量上再扫 margin。离线起点：eres 主判、camp 在 thr 下方约 ±0.08 救援（见 encoder 报告）。

过 P1a/P1b 且仍要做融合时：先小补丁 PresenceGate，再用**冻结的 P0 mix 波形**只改 decision 重跑 ASR（或 `--no-resume` 按新 decision）。未过门前禁止改默认。

**P1 门：** 真实 contest **稳定 ≥ P0 胜者 +0.005** 才改默认门控。

共享校准、只换 TSE 时用 `SKIP_CALIB=1`（本文件 P0/P1 **不要**换 TSE）。

---

## P2–P4（诊断 / 难例，非正式默认）

| 阶 | 做什么 | 现有入口 |
|----|--------|----------|
| P2 | enroll 纯净度 | 校准 `scores.jsonl` 的 `enroll_vad` meta；不必再跑全量 ASR |
| P3 | CMD 多次 sep | `score_encoders_on_sep.sh` 或已有 `ve_gate_cmp/sep_streams`；只对难例 uid |
| P4 | TSE 难例 | `LIMIT=… PIPELINE=sep_route` 或 ps4；**禁止**无门控全量替换 mix |

P5（换 ASR 模型）：不排期。

---

## T1–T4 下一刀（不改默认，直到真实 contest +0.005）

入口脚本：`./run_next_lift.sh t0|t1|t1b|t2|t3|t4`（须在 `/root/extract/ve`）。一次只动一个因子。主看 **CER=1 桶** 与 `contest`。

### T0 — 叠话加拒 / 冻结 τ（不重跑 ASR）

同一 `asr_cer` jsonl 上离线叠 `len_and_nontask_gray`。先 holdout τ，再冻结 zh 0.29305 / en 0.357868。

```bash
VE_OUT=/root/autodl-tmp/ve_mix_novad ./run_next_lift.sh t0
# → reports/lift_overlay/holdout_text.md  locked_text.md
```

`n_need_asr>0` 时冻结 τ 新放行的 pos 仍按 CER=1，contest 偏保守。

已有 mix 提取时导出提交文件（不重跑 pos ASR）：

```bash
VE_OUT=/root/autodl-tmp/ve_mix_novad ./run_next_lift.sh submit
# → $VE_OUT/reports/submit/result.json
```

### T1 — 同一 mix，只改 Qwen3 解码

固定锁定门控与提取波形。须先有 mix 提取（`samples.jsonl` + `all_results.jsonl`）。没有就先 `ENROLL_VAD=0 PIPELINE=mix … ./run_all.sh`。

```bash
# 在 /root/extract/ve（不要用 extract 根的 ve.sh 路径去 ve/ 里找）
VE_OUT=/root/autodl-tmp/ve_mix_novad ./run_next_lift.sh t1
# 或 ./ve.sh t1
# 或从 extract 根: VE_OUT=... ./ve.sh t1
# 或:
ASR_LANGUAGE=Chinese ./run_asr_cer.sh --out-dir "$VE_OUT/reports/asr_cer_zh"
ASR_LANGUAGE=Chinese ASR_DOMAIN_CONTEXT=1 ./run_asr_cer.sh --out-dir "$VE_OUT/reports/asr_cer_zh_domain"
```

对照：`language=auto`（现状） vs `Chinese` vs `Chinese`+领域 context（**不用唤醒词**）。  
**Go：** CER=1 接受桶下降且真实 contest ≥ 锁定 +0.005。

### T1b — 热词表 context（推荐于 T1 指令句）

Qwen3-ASR 官方 context 是 `Vocabulary: …`，不是「只转写那一句」。不强制 language。

```bash
VE_OUT=/root/autodl-tmp/ve_mix_novad ./run_next_lift.sh t1b
```

### T2 — CMD 滑窗时间选择

Presence = max_window；ASR = argmax 窗（两侧 80ms）。须重扫 τ，**另开** `ve_mix_win`，禁止覆盖 `ve_mix_novad`。已跑过：裁窗 ASR contest ≈ 0.554（No-Go）。

```bash
./run_next_lift.sh t2
```

### T2b — 滑窗 Presence + 整段 mix ASR

只改 ASR 输入：门控仍用 T2 滑窗分数，ASR 打原始 CMD 整段（不裁 0.8s）。有 `ve_mix_win` 时不重提取、不 `FORCE_CALIB`。

```bash
unset ASR_LANGUAGE ASR_DOMAIN_CONTEXT ASR_HOTWORDS
./run_next_lift.sh t2b
# → ve_mix_win/reports/asr_cer_fullmix/summary.md
```

无 T2 产物时才全量：`ASR_CROP=0` → `/root/autodl-tmp/ve_mix_win_fullasr`。  
RR 来自窗 τ，不要直接跟 novad 的 RR 比；主看 CER=1 相对 T2 裁窗是否回来，contest 能否 ≥ 锁定 +0.005。

### T3 — 时长不匹配才二次解码

```bash
VE_OUT=/root/autodl-tmp/ve_mix_novad ./run_next_lift.sh t3
# 或 ASR_RETRY_MISMATCH=1 ./run_asr_cer.sh --out-dir "$VE_OUT/reports/asr_cer_retry"
```

先 auto mix；不匹配则 Chinese+领域；若提取已是窗则再回退整段 mix。

### T4 — camp / 次优窗只否决

离线（本地可跑，用 `datasetA/sssss`）：

```bash
./run_next_lift.sh t4
# → datasetA/sssss/next_lift_eval.json
```

接入 extract（过门后）：`VETO_CAMP=1`，可选 `VETO_WINDOWS=1`。与叠话长句加拒合取。禁止救援 FN。  
**Go：** holdout test Δcontest ≥ 0.005 且额外 pos 误拒 ≤ 5。

---

## 禁止（与 PROBLEM.md §5 一致）

- 未过门把 `PIPELINE` 默认改成 ps4/wesep/sep_route
- `HOLDOUT_FRAC=0` 在全量上扫完 thr 当部署值并宣称泛化
- 用校准 JSON 的 contest 代替 `asr_cer` 写进结论
- P0/P1 过程中同时改 VAD + 编码器 + PIPELINE

---

## 记录模板

每跑完一轮在笔记里抄：

```text
VE_OUT=
ENROLL_VAD=  LANG_SPLIT=  PRESENCE_BACKEND=  HOLDOUT_FRAC=  FORCE_CALIB=
thr 文件=
RR=  FRR=  CER_total=  CER_accept=  contest=
相对 P0 Δcontest=
结论: Go / No-Go
```
