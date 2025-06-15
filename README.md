# 北航研究生课《AI学习框架与科学计算》大作业：基于RETFound的视网膜图像分类微调

本项目展示了如何使用 **PaddlePaddle** 深度学习框架，在一个名为 RETFound 的视觉基础模型（Vision Transformer-based foundation model）上进行微调，以完成四个不同视网膜图像数据集上的多分类任务。

## 1. 项目概览

本项目的核心目标是将一个大规模预训练模型（`RETFound_mae_meh`）适配到特定的医学影像领域，通过利用迁移学习，模型能够在相对较小的数据集上取得优异的性能。

- **模型**: `RETFound_mae`，一个基于 ViT-Large 架构的模型（`depth=24`, `embed_dim=1024`, `num_heads=16`），在一个海量的视网膜眼底图像数据集上，使用掩码自编码器（Masked Autoencoder, MAE）目标进行了预训练。
- **框架**: 飞桨 PaddlePaddle 3.0.0。
- **数据集**: 模型在四个公开的医学影像数据集上进行了微调和评估：
    - **OCTID**: 包含5个类别的光学相干断层扫描（OCT）图像数据集。
    - **Retina**: 包含4个类别的眼底图像数据集。
    - **APTOS2019**: 用于糖尿病视网膜病变分级的数据集，共5个等级。
    - **JSIEC**: 一个包含39个精细分类的综合眼科数据集。
- **主要特性**:
    - **分层学习率衰减 (Layer-wise Learning Rate Decay)**: 为不同深度的网络层设置不同的学习率，使微调过程更稳定高效。
    - **带 warmup 的余弦学习率调度策略**: 一种先进的学习率调整策略，在训练初期（如前10个周期）线性增加学习率，之后按余弦曲线衰减。
    - **混合精度训练**: 提高训练速度，同时减少显存占用。
    - **全面的评估指标**: 包括准确率、F1分数、ROC AUC和Kappa系数等。

## 2. 项目结构

代码库的组织结构如下：

```
.
├── main_finetune.py            # 运行微调和评估的主脚本
├── RETFound_MAE_finetune.ipynb # 用于交互式运行的Jupyter Notebook
├── engine_finetune.py          # 核心的训练与评估循环逻辑
├── models_vit.py               # ViT 模型定义 (RETFound_mae)
├── run.sh                      # 用于执行所有数据集实验的Shell脚本
├── requirements.txt            # Python 依赖项列表
│
├── util/
│   ├── datasets.py             # 数据集加载与数据增强工具
│   └── ...                     # 其他辅助脚本
│
└── output_dir/                 # 所有输出文件 (日志、评估指标、模型权重) 的存放目录
    └── ...
```

## 3. 环境配置与安装

### 环境要求
- Python 3.8+
- 支持 GPU 的 PaddlePaddle 3.0.0
- 一块支持 CUDA 的 NVIDIA GPU
- （可选）Jupyter Notebook 或 JupyterLab

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
    所有必需的数据集均可从原始 `RETFound_MAE` 代码库中找到并下载。
    - **数据集来源**: [https://github.com/rmaphoh/RETFound_MAE](https://github.com/rmaphoh/RETFound_MAE)

    请下载全部四个数据集（OCTID, Retina, APTOS2019, JSIEC），并按照以下结构组织文件夹。`run.sh` 脚本和 Notebook 默认会从项目的根目录寻找这些数据集文件夹。

    ```
    .
    ├── OCTID/
    │   ├── train/
    │   └── test/
    ├── Retina/
    │   ├── train/
    │   └── test/
    ... 以此类推
    ```

## 4. 如何使用

您可以通过两种主要方式来运行本项目的实验：

### 方式一：使用 Shell 脚本（推荐用于全量复现）
`run.sh` 脚本是复现所有实验的主要方式，为四个数据集依次执行微调和评估流程。

**运行全量实验：**
```bash
bash run.sh
```

### 方式二：使用 Jupyter Notebook（推荐用于快速查看训练过程与结果）
项目中的 `RETFound_MAE_finetune.ipynb` 文件将整个微调流程封装在了一个 Notebook 中。

**如何运行：**
1.  启动 Jupyter Notebook 服务器：`jupyter notebook`
2.  在浏览器中，打开 `RETFound_MAE_finetune.ipynb` 文件。
3.  在 Notebook 的第一个代码单元格中，您可以像命令行参数一样轻松地修改配置。
4.  按顺序执行所有单元格即可开始训练和评估。


## 5. 实验结果

模型在每个数据集上进行微调，并将在验证集上表现最好的模型权重用于最终的测试集评估，关键性能指标总结如下。

| 数据集 | 准确率 | F1 分数 (Macro) | ROC AUC (Macro) | Kappa 分数 |
| :--- | :---: | :---: | :---: | :---: |
| **OCTID** | 0.8966 | 0.8717 | 0.9845 | 0.8638 |
| **Retina** | 0.6354 | 0.5927 | 0.8490 | 0.4471 |
| **APTOS2019**| 0.8445 | 0.6819 | 0.9406 | 0.7615 |
| **JSIEC** | 0.8589 | 0.7805 | 0.9923 | 0.8523 |

*注：以上指标数据可在对应输出目录下的 `metrics_test.csv` 文件中找到。*

## 6. 致谢

本项目是在 **`rmaphoh/RETFound_MAE`** 的代码基础上完成的，我们对原作者的开源贡献表示衷心的感谢。

- **原始代码库**: [https://github.com/rmaphoh/RETFound_MAE](https://github.com/rmaphoh/RETFound_MAE)


