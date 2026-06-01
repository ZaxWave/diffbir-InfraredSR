# 实验一：SatVideoIRSDT 红外卫星视频超分

## 实验说明

- **任务**: Stage-2 ControlNet 扩散，无 Stage-1 重建模型
- **Condition**: 双三次上采样替代 SwinIR
- **数据集**: SatVideoIRSDT-redistributed（1154 张训练图）
- **基座模型**: Stable Diffusion 2.1（v2-1_512-ema-pruned）
- **配置**: configs/train/train_stage2.yaml
- **启动脚本**: train.sh

## 超参数

| 参数 | 值 |
|---|---|
| batch_size | 8 |
| learning_rate | 1e-4 |
| train_steps | 30000 |
| patch_size | 256 |
| noise_aug_timestep | 0 |
| ckpt_every | 5000 |
| image_every | 1000 |
| log_every | 50 |

## 产出

| 内容 | 路径 |
|---|---|
| Checkpoints | `checkpoints/*.pt` |
| TensorBoard | `./` (events.out.tfevents.*) |
| 日志 | `train_exp1_*.log`（项目根目录） |

## 断点续训

将 `configs/train/train_stage2.yaml` 中 `train.resume` 改为 checkpoint 路径，运行 `./train.sh`。

## 启动命令

```bash
./train.sh
```
