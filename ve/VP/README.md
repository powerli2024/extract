# VP：Voice Presence（main 分支历史参考）

> VP 不属于 `extract@sep` 的执行范围。sep 只产出 KWS 候选和 handoff；Presence 应在 `/root/extract@main` 运行。分支边界见 [仓库根 README](../../README.md)。

独立于 TSE 与 ASR。本目录可整体拆成单独仓库；当前复用上级 `VE/scripts` 的编码器与打分实现，避免复制权重加载逻辑。

## 问题（仅此一件）

已有较干净的目标说话人 **KWS enroll**，以及脏的 **CMD**（可能含他人、噪声、重叠）。  
运行时不知道正负。判断：**CMD 里有没有该说话人**。

| 输出 | 含义 |
|------|------|
| reject | 判不在（负样本希望如此） |
| accept | 判在（正样本希望如此） |

本模块**不算 CER、不跑 ASR、不做提取**。检测指标：RR / FRR / FAR，以及代理分 `0.5*RR+0.5*(1-FRR)`（**只用于 VP 内部排序**，不能当 VE 竞赛成绩）。

## 核心方法

```text
score = max_k cosine(embed(enroll'), embed(stream_k))
decision = score >= thr   → accept
```

实验要回答的不是「要不要余弦」，而是四个因子如何组合：

1. **enroll'**：是否 VAD  
2. **stream_k**：仅 mix / 一次分离 / 级联分离  
3. **embed**：eres / camp / resnet  
4. **thr**：全局 / 分语言；必须 **holdout** 选定  

实验设计全文见 [`DESIGN.md`](DESIGN.md)。

## 跑法

```bash
cd /path/to/VE/VP
# 可选 source ../.env_ve
./run_vp.sh --help
```

默认输出：`/root/autodl-tmp/vp/`（manifest + 各 cell 报告 + `reports/matrix.md`）。

## 与 VE 的关系

VE = VP 门控 +（默认 mix）ASR。VP 过门的配置再进 `VE/run_all.sh`。  
禁止把 VP 代理 contest 写成 VE 提交分。
