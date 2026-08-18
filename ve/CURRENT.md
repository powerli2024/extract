# 当前最优流水线（提交默认）

竞赛分：`0.5 * RR + 0.5 * (1 - CER)`。pos 误拒记 CER=1。  
**只拒「说话人不在」**；过门后默认 **mix ASR**，不要换成全量 TSE。

校内标定集约 **contest ≈ 0.739**（n=1838：pos 1364 / neg 474）。

## 运行时（AutoDL）

环境清单见 [`SETUP.md`](SETUP.md)。问题口径与已否决方向见 [`PROBLEM.md`](PROBLEM.md)。分阶命令见 [`EXPERIMENTS.md`](EXPERIMENTS.md)。

```bash
conda activate qwen3-asr
cd /root/extract/ve
git pull --ff-only
chmod +x *.sh
cp -n .env_ve.example .env_ve
./setup_env.sh && source .env_ve
ONLY=eres2netv2 ./download_presence_encoders.sh
./download_moss_onnx.sh          # Presence USE_SEP=1 需要
PIPELINE=mix ./check_env.sh

ENROLL_VAD=0 PIPELINE=mix \
PRESENCE_BACKEND=eres2netv2 USE_SEP=1 LANG_SPLIT=1 \
FORCE_CALIB=1 HOLDOUT_FRAC=0.3 \
./run_all.sh
```

下一刀（不改默认；须已有 mix 提取）：`VE_OUT=/root/autodl-tmp/ve_mix_novad ./run_next_lift.sh t1`

| 开关 | 值 | 原因 |
|------|----|------|
| `PIPELINE` | **mix** | 全量 ps4 / wesep / sep_route 真实 contest 都低于 mix |
| `PRESENCE_BACKEND` | eres2netv2 | 单模最佳；camp 灰带救援接到真实 ASR 为负 |
| `USE_SEP` | 1 | Presence 用 sep_once 打分；**提取仍用 mix** |
| `LANG_SPLIT` | 1 | zh/en 分 τ |
| `ENROLL_VAD` | **0** | 能量 VAD 真实 contest 略降（约 0.727→0.723） |

锁定 τ（eres + sep_once + lang_split，与 `datasetA/sssss` 标定一致）：

- zh：`0.29305`
- en：`0.357868`（按**唤醒词**语言，不是命令语言）

## 叠话加拒（只加拒、不救回）

余弦过门之后，若同时满足则改为拒识：

```text
0 ≤ score − τ ≤ 0.10  且  hyp 字数 ≥ 15  且  不是任务句
```

不要把低分 pos 因「像命令」救回。相对纯余弦约 **RR 0.865→0.886**，contest **0.728→0.739**。

脚本：`asr_probe/scripts/optimize_text_reject.py`（`len_and_nontask_gray`）。

## 不要进默认

- 全量 `PIPELINE=ps4|wesep|sep_route`
- 抬高 Presence τ
- 多编码器 OR / 灰带救援（代理 +0.012，真实 ASR 不涨）
- 把 Presence 代理 contest（CER:=FRR）当成提交分
- 换更大 ASR（P5 搁置）

ASR 下一刀：`./run_next_lift.sh t1|t2|t3|t4`（见 [`EXPERIMENTS.md`](EXPERIMENTS.md) T1–T4）。未过门（真实 contest +0.005）前不改上表默认。

- T1：同一 mix 上 `Chinese` / 领域 context（不用唤醒词）
- T2：CMD 滑窗 max cosine，ASR 用 argmax 窗
- T3：时长不匹配才二次解码，回退 mix
- T4：灰区 camp/次优窗只否决不救援

换更大 ASR（P5）仍搁置。
