# KWS 注册音频质量评估

`run_enroll_quality.sh` 在看到 CMD 之前评估注册音频，因此不依赖 DatasetA、正负标签或“这个 CMD 是否有注册人”的先验。

## 输出口径

每条音频输出以下可审计指标：

- 有效时长、能量语音占比和静音占比
- RMS/峰值电平、削波比例、直流偏置、非有限采样
- 帧能量对比、95% 占用带宽
- 默认 ERes2NetV2 下，原音对 VAD 裁切和 20 dB 轻噪声的声纹 embedding 稳定性
- 若 manifest 已有注册唤醒词 ASR CER，也把它作为**注册侧已知信息**计入；不会读取 CMD 文本

原始指标是主结果。`quality_score` 只是默认策略的透明汇总，不能宣称等于后续 FAR/FRR。决策分三级：

- `pass`：技术指标全部通过
- `review`：建议人工听检或再录一条
- `reject`：静音、过短、有效语音不足、严重削波或异常电平等确定性硬问题，建议复录

带宽仅作为原始观测值，不改变默认决策；短唤醒词的音素组成会让频谱带宽发生很大变化。声纹稳定性只触发 `review`，必须经过独立的带标签混合 CMD 验证后，才能升级为硬门。接近时长下限的音频同样只复核，避免因几毫秒的 VAD 波动直接拒绝。

## 运行

```bash
cd /root/extract-main/ve
source .env_ve

# run_all 已生成 manifest 时
./run_enroll_quality.sh \
  --manifest "$VE_OUT/manifest/enrollment_manifest.jsonl" \
  --out-dir "$VE_OUT/reports/enroll_quality"

# 单条注册音频
./run_enroll_quality.sh \
  --audio /path/to/enroll.wav \
  --out-dir /tmp/enroll_quality_one

# CPU 快速信号质检，不加载声纹模型
./run_enroll_quality.sh \
  --audio-dir /path/to/enrolls \
  --backend none \
  --out-dir /tmp/enroll_quality_signal
```

产物：`enroll_quality.jsonl` 和 `summary.json`。加 `--fail-on-reject` 后，只要存在硬拒绝就以非零状态退出，适合注册服务的质量门。

## 校准原则

默认阈值在代码的 `QualityPolicy` 中，也可用 `--policy policy.json` 覆盖。上线前应在**独立于 DatasetA**的真实注册录音上冻结策略：至少包含安静、远场、削波、过低电平、电视背景声、双人叠话和窄带音频。策略一旦冻结，测试 CMD 只能经过同一个推理入口，不能按测试集标签改阈值。

## 两层评价

本模块解决上线时的无标签质检。离线比较不同注册提纯方案时，应另用与最终测试隔离的带标签 CMD：正 CMD 测目标声纹保持，负 CMD 测冒认，报告 EER/AUC、固定 FAR 下 FRR、固定 FRR 下 FAR及 bootstrap 置信区间。带标签 CMD 更接近最终目标，但只能用于开发集校准和方案验收，不能用于逐条选择最终测试样本的注册音轨。
