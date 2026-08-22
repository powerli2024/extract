# extract `sep` — KWS 分离实验专用分支

仓库：https://github.com/powerli2024/extract.git  
**本分支只做 KWS BSS + Qwen CER oracle 选轨。不做 Presence / mix ASR。**  
竞赛 VE 在 [`main`](https://github.com/powerli2024/extract/tree/main) 的 `ve/`。

下游选轨 / 条件 SE / Presence 否决在独立仓  
https://github.com/powerli2024/kws.git （读本分支的 `VM_OUT`）。

```text
datasetA kws
  → extract@sep  ./run_sep.sh          # s1–s8，无 MMS-FA
  → $VM_OUT/{pos,neg}/s*/index.jsonl   # 每轨 CER
  → $VM_OUT/best_sep/index.jsonl       # 跨阶段 CER oracle
  → $VM_OUT/kws_handoff.json           # 契约
  → kws  rebuild_best_sep / T0–T4
```

契约见 `KWS_HANDOFF.md`。选轨：**阶段内** argmin CER，平局保 `original`；**跨阶段** argmin CER，平局优先非 original（与现网 `best_sep` 收集一致）。

## AutoDL

```bash
cd /root
git clone -b sep https://github.com/powerli2024/extract.git extract
cd /root/extract && chmod +x *.sh run_sep.sh
export DATA_DIR=/root/autodl-tmp/datasetA
export VM_OUT=/root/autodl-tmp/kws_sep
export MOSS_CKPT_DIR=/root/autodl-tmp/checkpoints
export ASR_MODEL_DIR=/root/autodl-tmp/Qwen3-ASR-1.7B
./setup_env.sh && source ./env.sh
./download_models.sh
./check_env.sh
./run_sep.sh --limit 20
./run_sep.sh
# 全量重跑: ./run_sep.sh --force
```

然后在 kws 仓：

```bash
python scripts/rebuild_best_sep.py --pos-neg /root/autodl-tmp/kws_sep
python scripts/analyze_dual_zero.py --pos-neg /root/autodl-tmp/kws_sep
```

默认阶段：`collect,s1,s2,s3,s4,s5,s6,s7,s8,compare,eval,handoff`。  
`./ve.sh` 在本分支会直接退出。

## 环境与权重

与 `main` 相同：数据、ONNX/ClearVoice、Qwen3 只放 `/root/autodl-tmp`。  
`./setup_env.sh` → `./download_models.sh` → `./check_env.sh`。

| 阶段 | 含义 |
|------|------|
| s1 | ONNX 一阶 original/spk1/spk2 |
| s2 | ClearVoice 一阶 |
| s3 | ONNX cascade（输入同 split s1 wav） |
| s4 | ClearVoice cascade（输入 s2） |
| s5–s8 | 门控二次（抢救，不是默认 enroll） |
| handoff | 写出 `best_sep/` + `kws_handoff.json` |
