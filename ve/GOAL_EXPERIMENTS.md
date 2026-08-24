# 以最终分数为目标的门控、TSE 与 ASR 实验

唯一上线指标：

```text
score = (RR_neg + 1 - CER_pos_micro) / 2
```

正样本被拒、ASR 失败或缺失均按该条 `CER=1`。`final_evaluate.py --strict` 是唯一正式计分器；固定 FAR、FRR 和仅接受样本 CER 都只是诊断指标。

## 1. 冻结输入与基线

先设置真实路径，`BEST_SEP_DIR` 必须是含 `pos/`、`neg/`、`index.jsonl` 的内层目录：

```bash
cd /root/extract-main/ve
export DATA_DIR=/root/autodl-tmp/datasetA
export BEST_SEP_DIR=/root/autodl-tmp/你的注册音频内层目录
export DEVICE=cuda:0
export ASR_MODEL_DIR=/root/autodl-tmp/Qwen3-ASR-1.7B
```

先复现冻结基线（max 门控、MossFormer 声纹选路、无注册 VAD、无历史 ASR）：

```bash
VE_OUT=/root/autodl-tmp/ve_goal/baseline_sep_route \
PIPELINE=sep_route LOCKED_THR=1 STREAM_POLICY=max ENROLL_VAD=0 \
STRICT_ENROLL=1 STRICT_EVAL=1 ASR_RESUME=0 ASR_RETRY_MISMATCH=1 \
EXTRA_REJECT=0 LIMIT=0 ./run_all.sh
```

要求 `reports/final_eval/summary.json` 中 coverage errors 为 0；否则本轮不可参与排名。

## 2. 用所有正样本的真实 mix ASR 优化门控

该实验忽略旧门控，强制将每个正样本的原始 CMD mix 送入同一个 ASR。因此降低阈值后新增放行的正样本也有真实编辑距离，不再用 FRR 冒充 CER。

```bash
ALL_POS_MIX_OUT=/root/autodl-tmp/ve_goal/all_pos_mix \
STRICT_ENROLL=1 ASR_RETRY_MISMATCH=1 LIMIT=0 \
./run_all_pos_mix_asr.sh

python scripts/optimize_gate_for_score.py \
  --decisions /root/autodl-tmp/ve_goal/baseline_sep_route/results/all_results.jsonl \
  --asr-all-pos /root/autodl-tmp/ve_goal/all_pos_mix/reports/asr_cer/asr_results.jsonl \
  --baseline-thr configs/locked_thr.json \
  --out-dir /root/autodl-tmp/ve_goal/gate_score_opt \
  --policies max,strict_rescue,mix --holdout-frac 0.30 --seeds 100 --strict
```

只有 holdout 均值更高、相对冻结基线的置信区间下界不为负、且独立批次复现时，才把候选阈值写入新的配置文件。`full_data_diagnostic` 是 in-sample 上界，禁止直接上线。当前 `strict_rescue` 的固定参数为 `high=0.08/floor=0.10/dominance=0.05`，没有复用 max 阈值。

## 3. mix、分离选路与 TSE 的同门控全量比较

先准备 WeSep 和 PS4；Cond-TasNet 只有已有域内训练权重时才有意义：

```bash
./download_wesep.sh
./download_models.sh
export COND_TASNET_CKPT=/root/autodl-tmp/ve_models/cond_tasnet/best.pt  # 没有则从 PIPELINES 删除
```

先冒烟；每个失败臂会记录 `arm_status.txt`，不会阻断其他臂：

```bash
EXP_ROOT=/root/autodl-tmp/ve_goal/smoke \
PIPELINES=mix,sep_route,adaptive_route,wesep,ps4 \
LIMIT=64 SKIP_ASR=0 ./run_compare_pipelines.sh
```

冒烟通过后全量跑。所有臂使用同一冻结门控，差异只来自送给 ASR 的波形：

```bash
EXP_ROOT=/root/autodl-tmp/ve_goal/pipelines \
PIPELINES=mix,sep_route,adaptive_route,wesep,ps4,cond_tasnet \
LIMIT=0 LOCKED_THR=1 STRICT_EVAL=1 ASR_RESUME=0 EXTRA_REJECT=0 \
./run_compare_pipelines.sh
```

汇总文件是 `/root/autodl-tmp/ve_goal/pipelines/pipelines_summary.md`，按严格 `final_eval` 分数排序并显示 coverage errors。不能用 `CER_accept` 选上线方案，因为它没有计入误拒。

## 4. mix ASR 与盲分离各轨 ASR 诊断

`run_compare_pipelines.sh` 回答“整条上线管线谁得分高”；下面的 probe 回答“MossFormer 的哪条轨损坏/改善了文本”，并覆盖 pos、neg 及混有其他人的 CMD：

```bash
# 先用已有 Presence 运行保存 d1 两轨；若尚无 sep_streams：
python scripts/run_extract.py \
  --samples /root/autodl-tmp/ve_goal/baseline_sep_route/manifest/samples.jsonl \
  --out-dir /root/autodl-tmp/ve_goal/presence_cache \
  --thr-file configs/locked_thr.json --presence-backend eres2netv2 \
  --use-sep --sep-depth 1 --save-sep-wavs --skip-tse --no-enroll-vad \
  --stream-policy max --device cuda:0

SAMPLES=/root/autodl-tmp/ve_goal/baseline_sep_route/manifest/samples.jsonl \
SEP_ROOT=/root/autodl-tmp/ve_goal/presence_cache/sep_streams \
ASR_PROBE_OUT=/root/autodl-tmp/ve_goal/asr_probe \
ARMS=no_sep,sep_once LIMIT=0 ./asr_probe/run_asr_probe.sh
```

若要评估二次级联，再重新保存 d2 并加入 `sep_multi`；不要仅因“所有轨中最小 CER”好看就上线，因为线上不知道哪轨 CER 最小。

## 5. 上线门槛

候选必须同时满足：

1. 同一冻结 manifest、相同 ASR 配置、无历史结果复用；严格覆盖错误为 0。
2. 全量 `score` 高于冻结基线，并在按注册人/语言/噪声条件分层的 paired bootstrap 中 95% 下界大于 0。
3. 分别报告纯净、噪声、单非目标说话人、目标与他人重叠四类；负样本 FAR 不得由某一类集中恶化。
4. 测量端到端 p50/p95 延迟、峰值显存和失败率。TSE 只在分数增益覆盖资源代价时上线。
5. 先全局选择 `mix` 或一个 TSE；按样本动态路由必须使用与标签/CER无关的在线特征，并在独立 holdout 上训练和冻结。

注册音频质检继续作为诊断特征。可以用 CMD 与正负标签离线衡量它是否能预测 FRR/FAR/CER，但不能把标签或参考文本带入线上，也不能因为低质量而直接拒绝全部 CMD；后者会把所有正样本记为 `CER=1`。

## 6. 后续模型优先级

1. **[Personal/Target-speaker VAD](https://research.google/pubs/personal-vad-speaker-conditioned-voice-activity-detection/)**：直接输出逐帧“目标、非目标、非语音”概率，更贴合混有其他人声和噪声的 Presence 问题，优先级高于继续堆更多 utterance-level 最大余弦。
2. **[WeSep pBSRNN/pDPCCN](https://github.com/wenet-e2e/wesep)**：已接入预训练 WeSep 臂；若英文 VoxCeleb 权重域差明显，应使用本域合成重叠、噪声、混响数据微调，再与 mix 比。
3. **[USEF-TSE / USEF-TFGridNet](https://github.com/ZBang/USEF-TSE)**：可作研究臂，但公开权重为 8 kHz 且仓库为 CC BY-NC 4.0；必须先解决 16→8→16 kHz 与商业许可，不应直接替换生产链路。
4. **[Target-speaker ASR（如 DiCoW）](https://github.com/BUTSpeechFIT/TS-ASR-Whisper)**：将目标说话人活动概率直接条件化给 ASR，避免“先分离再 ASR”的伪影；属于需要训练/适配的新路线，先在现有数据上作离线研究臂。
5. **[WavLM speaker verification 表征](https://github.com/microsoft/unilm/tree/master/wavlm)**：可作为第二声纹编码器或 PVAD 条件表征，但必须独立校准，不能与 ERes2NetV2 共用阈值。
