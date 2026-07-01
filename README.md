# 编程实战作业 2：扩散模型生成 MNIST 图像

本项目用于完成课堂“扩散模型”编程实战作业。基于 PyTorch 实现一个简洁 DDPM，在 MNIST 数据集上训练生成手写数字，并计算 FID 作为效果指标。

## 任务对应关系

- 方案：DDPM 扩散模型
- 数据集：MNIST
- 框架：PyTorch
- 输出：训练日志、生成样例图、FID 指标、模型权重、可上传 GitHub 的完整代码

## 目录结构

```text
diffusion_homework_project/
├── train_ddpm.py              # 训练 DDPM
├── sample_ddpm.py             # 加载模型并生成图片
├── evaluate_fid.py            # 计算 FID
├── models/
│   └── unet.py                # 简洁 U-Net 噪声预测网络
├── utils/
│   ├── diffusion.py           # DDPM 前向加噪与反向采样
│   ├── data.py                # MNIST 数据加载
│   ├── fid.py                 # FID 计算工具
│   └── plot.py                # 曲线绘制
├── outputs/                   # 训练输出目录
└── requirements.txt
```

## 环境安装

```bash
cd diffusion_homework_project
pip install -r requirements.txt
```

建议使用 GPU。CPU 也能运行，但训练较慢。

## 1. 训练 DDPM

快速测试：

```bash
python train_ddpm.py --epochs 1 --batch-size 128 --timesteps 200
```

正式训练建议：

```bash
python train_ddpm.py --epochs 30 --batch-size 128 --timesteps 1000 --lr 2e-4
```

训练完成后会生成：

```text
outputs/ddpm_mnist/
├── train_log.csv
├── loss_curve.png
├── ckpt_best.pt
├── ckpt_last.pt
└── samples_epoch_xxx.png
```

## 2. 生成图片

```bash
python sample_ddpm.py --ckpt outputs/ddpm_mnist/ckpt_best.pt --num-samples 64
```

输出：

```text
outputs/ddpm_mnist/generated_grid.png
outputs/ddpm_mnist/generated_images/*.png
```

## 3. 计算 FID

```bash
python evaluate_fid.py --ckpt outputs/ddpm_mnist/ckpt_best.pt --num-samples 1000
```

输出：

```text
outputs/ddpm_mnist/fid_report.txt
```

说明：为了适配课堂作业和轻量运行，本项目默认使用 MNIST 训练的轻量 CNN 特征提取器计算 FID-like 指标。该指标适合本作业内部比较，但不等同于 ImageNet Inception-v3 标准 FID。

## 4. 作业提交建议

提交内容建议包括：

1. 终端训练日志截图：运行 `train_ddpm.py` 的终端截图。
2. Loss 曲线：`outputs/ddpm_mnist/loss_curve.png`。
3. 生成结果：`generated_grid.png` 或 `samples_epoch_xxx.png`。
4. FID 报告：`fid_report.txt`。
5. GitHub 仓库：上传本项目完整代码。

## 常用参数

```bash
python train_ddpm.py \
  --epochs 30 \
  --batch-size 128 \
  --lr 2e-4 \
  --timesteps 1000 \
  --base-channels 64 \
  --out-dir outputs/ddpm_mnist
```

如果显存不足，可降低：

```bash
--batch-size 64 --base-channels 32
```
