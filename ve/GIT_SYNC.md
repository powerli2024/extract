# 本地 ↔ GitHub ↔ AutoDL 同步

VE 已嵌在 **extract** 仓的 `ve/`（不含 dataset / 模型 / 跑数）。  
数据与权重仍放 `/root/autodl-tmp/`。

仓库：https://github.com/powerli2024/extract.git  
AutoDL 克隆目录：`/root/extract`。竞赛流水线在 `/root/extract/ve`。KWS 分离在分支 **`sep`**。

## 一次配置

### 本地（Windows）

代码在 `d:\media\VM`（extract 仓）。VE 在 `d:\media\VM\ve`。独立的 `d:\media\VE` 只作历史对照，不要再往那边 push。

```powershell
cd d:\media\VM
git remote -v   # origin → powerli2024/extract.git
git add ve ve.sh README.md
git status      # 确认没有 .env_ve、模型、wav
git commit -m "Nest VE under extract/ve for AutoDL."
git push origin main
```

### AutoDL

已有 `/root/extract` 时：

```bash
cd /root/extract
git pull --ff-only
chmod +x ve.sh ve/*.sh
cp -n ve/.env_ve.example ve/.env_ve
cd ve
```

首次若还没有仓：

```bash
cd /root
git clone https://github.com/powerli2024/extract.git extract
cd /root/extract && chmod +x ve.sh ve/*.sh
```

MossFormer 封装在同仓 `../scripts/mossformer2_onnx.py`（extract 根 `scripts/`），不必再旁路挂载 `/root/media/VM`。

## 日常更新

**本地改完推送：**
```powershell
cd d:\media\VM
git add -A
git commit -m "简述改动"
git push origin main
```

**AutoDL 拉取：**
```bash
cd /root/extract
git pull --ff-only
chmod +x ve.sh ve/*.sh
# 若有本地改过的 .sh 冲突：git stash -u && git pull --ff-only && git stash pop
```

下一刀（须已有 mix 提取 + Qwen3）：

```bash
cd /root/extract/ve
VE_OUT=/root/autodl-tmp/ve_mix_novad ./run_next_lift.sh t1
# 或从 extract 根: ./ve.sh t1
```

## 注意

| 进 git | 不进 git |
|--------|----------|
| `ve/scripts/`、`ve/*.sh`、`README`、`.env_ve.example` | `.env_ve`、模型、`ve_*` 跑数、`datasetA` |

不要把 `/root/autodl-tmp/ve_*` 放进仓库。
