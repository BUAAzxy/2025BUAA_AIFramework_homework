# 北航研究生课《AI学习框架与科学计算》大作业：基于RETFound的视网膜图像分类微调

本项目展示了如何使用 **PaddlePaddle** 深度学习框架，在一个名为 RETFound 的视觉基础模型（Vision Transformer-based foundation model）上进行微调，以完成四个不同视网膜图像数据集上的多分类任务。

## 1. 项目概览

本项目的核心目标是将一个大规模预训练模型（`RETFound_mae_meh`）适配到特定的医学影像领域。通过利用迁移学习，模型能够在相对较小的数据集上取得优异的性能。

- **模型**: `RETFound_mae`，一个基于 ViT-Large 架构的模型。它在一个海量的视网膜眼底图像数据集上，使用掩码自编码器（Masked Autoencoder, MAE）目标进行了预训练。
- **框架**: 飞桨 PaddlePaddle 3.0.0。
- **数据集**: 模型在四个公开的医学影像数据集上进行了微调和评估：
    - **OCTID**: 包含5个类别的光学相干断层扫描（OCT）图像数据集。
    - **Retina**: 包含4个类别的眼底图像数据集。
    - **APTOS2019**: 用于糖尿病视网膜病变分级的数据集，共5个等级。
    - **JSIEC**: 一个包含39个精细分类的综合眼科数据集。
- **主要特性**:
    - **分层学习率衰减 (Layer-wise Learning Rate Decay)**: 为不同深度的网络层设置不同的学习率，使微调过程更稳定高效。
    - **带 warmup 的余弦学习率调度策略**: 一种先进的学习率调整策略。
    - **混合精度训练**: 提高训练速度，同时减少显存占用。
    - **全面的评估指标**: 包括准确率、F1分数、ROC AUC和Kappa系数等。

## 2. 项目结构

代码库的组织结构如下：

```
.
├── main_finetune.py      # 运行微调和评估的主脚本
├── engine_finetune.py    # 核心的训练与评估循环逻辑
├── models_vit.py         # ViT 模型定义 (RETFound_mae)
├── run.sh                # 用于执行所有数据集实验的Shell脚本
├── requirements.txt      # Python 依赖项列表
│
├── util/
│   ├── datasets.py       # 数据集加载与数据增强工具
│   ├── lr_decay.py       # 分层学习率衰减的实现
│   ├── lr_sched.py       # 学习率调度逻辑
│   ├── misc.py           # 其他辅助工具 (分布式训练、日志、模型保存等)
│   └── pos_embed.py      # ViT 位置编码插值
│
└── output_dir/           # 所有输出文件 (日志、评估指标、模型权重) 的存放目录
    ├── RETFound_mae_meh-OCTID/
    ├── RETFound_mae_meh-Retina/
    ├── RETFound_mae_meh-APTOS2019/
    └── RETFound_mae_meh-JSIEC/
        ├── log.txt
        ├── metrics_test.csv
        ├── metrics_val.csv
        └── checkpoint-best.pdparams
```

## 3. 环境配置与安装

### 环境要求
- Python 3.8+
- 支持 GPU 的 PaddlePaddle 3.0.0
- 一块支持 CUDA 的 NVIDIA GPU

### 安装步骤
1.  **克隆代码库:**
    ```bash
    git clone <repository-url>
    cd 2025BUAA_AIFramework_homework
    ```

2.  **安装依赖:**
    建议在虚拟环境（virtual environment）中执行以下命令。
    ```bash
    pip install -r requirements.txt
    ```

3.  **准备数据集:**
    请下载全部四个数据集（OCTID, Retina, APTOS2019, JSIEC），并按照以下结构组织文件夹。`run.sh` 脚本默认会从项目的根目录寻找这些数据集文件夹。

    ```
    .
    ├── OCTID/
    │   ├── train/
    │   │   ├── class_1/
    │   │   └── ...
    │   ├── val/
    │   └── test/
    │
    ├── Retina/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ... 以此类推，组织 APTOS2019 和 JSIEC
    ```

## 4. 如何使用

`run.sh` 脚本是复现所有实验的主要方式。它会为四个数据集依次执行微调和评估流程。

**运行全量实验：**
```bash
bash run.sh
```

**在单个数据集上运行微调：**
你也可以从 `run.sh` 中提取单个命令来针对特定数据集进行实验。例如，在 OCTID 数据集上运行：
```bash
python main_finetune.py \
  --data_path OCTID \
  --finetune RETFound_mae_meh \
  --epochs 50 --warmup_epochs 10 \
  --batch_size 16 --lr 5e-4 --layer_decay 0.65 \
  --task RETFound_mae_meh-OCTID --nb_classes 5
```

### 关键参数说明
- `--data_path`: 数据集文件夹的路径 (例如, `OCTID`)。
- `--finetune`: 要从 Hugging Face Hub 下载的预训练模型名称 (`RETFound_mae_meh`)。
- `--epochs`: 总训练轮数。
- `--warmup_epochs`: 学习率预热（warmup）阶段的轮数。
- `--batch_size`: 每块GPU上的训练批量大小。
- `--lr`: 峰值学习率。
- `--layer_decay`: 应用于分层学习率衰减的衰减因子。
- `--task`: 为本次实验设定的唯一名称，将用于创建输出目录。
- `--nb_classes`: 数据集的类别数量。

## 5. 实验结果

模型在每个数据集上进行微调，并将在验证集上表现最好的模型权重用于最终的测试集评估。关键性能指标总结如下。

| 数据集 | 准确率 | F1 分数 (Macro) | ROC AUC (Macro) | Kappa 分数 |
| :--- | :---: | :---: | :---: | :---: |
| **OCTID** | 0.8966 | 0.8717 | 0.9845 | 0.8638 |
| **Retina** | 0.6354 | 0.5927 | 0.8490 | 0.4471 |
| **APTOS2019**| 0.8445 | 0.6819 | 0.9406 | 0.7615 |
| **JSIEC** | 0.8589 | 0.7805 | 0.9923 | 0.8523 |

*注：以上指标数据提取自对应输出目录下的 `metrics_test.csv` 文件。*

## 6. 代码说明

- **`main_finetune.py`**: 该脚本是整个项目的配置中心和启动器。它负责解析命令行参数，初始化分布式训练环境，加载 `RETFound_mae` 模型，并处理预训练权重的适配（包括位置编码插值）。在启动主训练循环之前，它还会设置好数据加载器、优化器和损失缩放器。

- **`engine_finetune.py`**: 该文件包含了单个训练周期 (`train_one_epoch`) 和完整评估过程 (`evaluate`) 的核心逻辑。训练函数管理了支持混合精度的前向/反向传播、梯度累积和日志记录。评估函数则负责计算一套全面的性能指标（准确率、F1、ROC AUC、精确率、召回率、Kappa等），并将结果保存到CSV文件，同时为测试集生成混淆矩阵图。

- **`models_vit.py`**: 定义了 `VisionTransformer` 的模型结构。其中的 `RETFound_mae` 工厂函数创建了一个 ViT-Large 模型（`depth=24`, `embed_dim=1024`, `num_heads=16`），这是本项目的骨干网络。它提供了多头自注意力、MLP块和图像块嵌入等标准化组件的实现。

- **`util/`**: 该目录包含一系列辅助模块：
    - `datasets.py`: 构建用于训练（包含 `RandAugment` 等数据增强）和验证/测试（标准缩放和裁剪）的数据预处理流水线。它使用 `paddle.vision.datasets.DatasetFolder` 来加载图像数据。
    - `lr_decay.py`: 实现了分层学习率衰减（Layer-wise Learning Rate Decay），该技术为网络中较浅的层设置较小的学习率，为较深的层设置较大的学习率，从而稳定深度模型的微调过程。
    - `lr_sched.py`: 提供了一个根据带有线性预热（warmup）的余弦衰减策略来调整学习率的函数。这是训练现代Transformer模型的标准实践。
    - `pos_embed.py`: 包含一个关键的位置编码插值工具。当微调时使用的图像分辨率与预训练时不同，该工具可以通过缩放位置编码网格来解决适配问题。
    - `misc.py`: 一系列工具函数的集合，用于处理分布式训练、记录平滑后的指标、以及以PaddlePaddle格式保存/加载模型和优化器状态。
