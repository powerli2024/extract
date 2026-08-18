# USEF-TSE 直接使用时的采样率策略

公开权重（[ZBang/USEF-TSE](https://github.com/ZBang/USEF-TSE)）配置为 **`sample_rate: 8000`**。  
VE 数据、Presence、ASR 均为 **16 kHz**。若直接加载预训练 ckpt（不重新训练），必须显式处理 16↔8。

## 信号链（推荐）

```text
Presence @ 16 kHz（不变）
        │
        ▼ accept
mix16 / enroll16
   │              │
   ▼              ▼
resample↓      resample↓     ← 同一 method、同一抗混叠
mix8 / enroll8
   │
   ▼
USEF-TSE @ 8 kHz
   │
   ▼
est8 ──resample↑──► est16 ──► ASR（16 kHz）
```

要点：
- **Presence 不要跟到 8 kHz**，否则 thr / VAD / lang_split 全部失效或需重校准。
- enroll 与 mix **必须同一降采样实现**（相位与带宽一致，否则条件信息错位）。
- 升采样只是给 ASR 对齐时钟，**不能恢复** 4 kHz 以上已丢频谱。

## 降采样方法比较

| 方法 | 抗混叠 | 2:1 精度 | 建议 |
|------|--------|----------|------|
| `scipy.signal.resample_poly(x, 1, 2)` | 有（FIR） | 有理倍数，精确 | **首选** |
| soxr / `librosa.resample(..., res_type="soxr_hq")` | 强 | 好 | 首选并列 |
| `librosa` `kaiser_best` | 强 | 好 | 可用 |
| `librosa` `kaiser_fast` | 中 | 好 | 冒烟加速 |
| `torchaudio.functional.resample` | 依赖 quality 参数 | 好 | GPU 路径可用 |
| 隔点抽取 / 无低通抽稀 | **无** | — | **禁止** |

实现时在 `audio_io.resample_wav(..., method=)` 中锁定默认 `poly` 或 `soxr_hq`，并把 `method` 写入 extract 结果 JSON，保证可复现对照。

## 对竞赛指标的影响

1. **带宽**：Nyquist 4 kHz → 擦音、齿音、部分中文辅音变钝 → **ASR CER 往往变差**（即便 SI-SDR 在英文双人混合集上好看）。
2. **域差**：权重来自 WSJ0-2mix / WHAM(R) **英文** 8 kHz；datasetA 中英文短指令 + 噪声条件不同。
3. **与 mix 基线**：mix 走全带 16 kHz。USEF 若只在「已 accept」子集上略好、全量 contest 仍输 mix，则不应改默认 `PIPELINE`。
4. **假阳性代价**：Presence 放宽后，坏提取进 ASR 会把 CER 拉到 1 附近；8 kHz 伪影可能放大该效应。

## 接入前实验清单（LIMIT）

在同一 Presence 决策（同一 `VE_OUT` thr / VAD）下：

| 臂 | 内容 |
|----|------|
| A | `PIPELINE=mix` @16k |
| B | USEF + `resample_poly` 16→8→16 |
| C | USEF + `soxr_hq` |
| D（可选） | 仅对 CMD 降采样、enroll 另法（预期应更差，作负对照） |

报告：accept 子集 CER、全量 contest、耗时/显存。  
仅当 B 或 C 在 holdout/验证意义上稳定优于 A 时，再把 `PIPELINE=usef` 做成默认候选。

## 许可

官方 **CC BY-NC 4.0**。研究/竞赛实验需自行确认规则是否允许。

## 状态

`PIPELINE=usef` 尚未接入；`./run_all.sh --help` 中有摘要。确认骨干（TFGridNet / SepFormer）与权重域（wsj0 / wham / whamr）后再实现。
