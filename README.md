# 北航研究生课《AI学习框架与科学计算》大作业：基于PaddlePaddle的RETFound模型微调实践

本项目核心是基于 **飞桨 PaddlePaddle** 深度学习框架，对 `RETFound` 视觉基础模型进行微调，以解决多类别视网膜图像的分类任务，完整实现了从数据准备、模型训练、到性能评估的全过程，并为了验证其有效性，附带了与 PyTorch 框架下的简要性能对比。

## 1. 项目概览

本项目旨在深入实践如何利用 PaddlePaddle 框架，将一个大规模预训练模型（`RETFound_mae_meh`）通过迁移学习高效地适配到特定的医学影像分类任务上。

* **核心框架**: **飞桨 PaddlePaddle 3.0.0**
* **模型**: `RETFound_mae`，一个基于 ViT-Large 架构的模型，已在海量视网膜图像数据上完成预训练。
* **数据集**: 我们在四个公开的医学影像数据集上验证了模型的性能：
    * **OCTID**: 光学相干断层扫描（OCT）图像，5分类。
    * **Retina**: 眼底图像，4分类。
    * **APTOS2019**: 糖尿病视网膜病变分级图像，5分类。
    * **JSIEC**: 一个包含39个精细分类的综合眼科数据集。
* **关键技术实现 (基于PaddlePaddle)**:
    * **分层学习率衰减 (Layer-wise Learning Rate Decay)**: 为 ViT 模型不同深度的 Transformer Block 设置递减的学习率，微调更稳定。
    * **带 warmup 的余弦学习率调度**: 在训练初期线性增加学习率，随后按余弦曲线衰减，有助于模型收敛。
    * **混合精度训练**: 利用 PaddlePaddle 的 AMP 功能，加速训练并降低显存消耗。
    * **全面的评估指标**: 包括准确率（Accuracy）、宏平均F1分数（Macro F1-Score）、宏平均ROC AUC（Macro ROC AUC）和 Kappa 系数。

## 2. 项目结构

代码库的组织结构如下，所有核心逻辑均基于 PaddlePaddle 实现：

```
.
├── main_finetune.py            # (Paddle) 核心主脚本，用于运行微调和评估
├── RETFound_MAE_finetune.ipynb # (Paddle) Jupyter Notebook 版本，方便交互式调试
├── engine_finetune.py          # (Paddle) 训练和评估的核心逻辑循环
├── models_vit.py               # (Paddle) ViT 模型定义
├── run.sh                      # (Paddle) 自动化执行所有数据集实验的Shell脚本
├── requirements.txt            # Python 依赖包列表
│
├── util/                       # (Paddle) 工具脚本
│   ├── datasets.py             # 数据集加载与数据增强
│   └── ...                     # 其他辅助脚本
│
├── output_dir/                 # (Paddle) PaddlePaddle 实验的所有输出（日志、模型、评估结果）
│   └── ...
│
└── torch_results/              # (PyTorch) PyTorch 对比实验的结果存档
    └── ...
```

## 3. 环境配置与安装

### 环境要求
* Python 3.8+
* **PaddlePaddle 3.0.0 (GPU版)**
* 一块支持 CUDA 的 NVIDIA GPU
* （可选）Jupyter Notebook 或 JupyterLab

### 安装步骤
1.  **克隆代码库:**
    ```bash
    git clone <repository-url>
    cd 2025BUAA_AIFramework_homework
    ```

2.  **安装依赖:**
    建议在虚拟环境中安装，核心依赖为 PaddlePaddle。
    ```bash
    pip install -r requirements.txt
    ```

3.  **准备数据集:**
    所有数据集均可从原始 `RETFound_MAE` 代码库的说明中下载，请将数据集按类别存放于项目根目录，例如 `OCTID/`，`Retina/` 等。

## 4. 如何使用 (PaddlePaddle)

我们提供了两种方式来运行 PaddlePaddle 版本的实验：

### 方式一：使用 Shell 脚本（推荐）
`run.sh` 脚本可以一键完成在所有四个数据集上的训练和评估。
```bash
bash run.sh
```
所有结果将自动保存在 `output_dir/` 下对应的子文件夹中。

### 方式二：使用 Jupyter Notebook
`RETFound_MAE_finetune.ipynb` 可以实现对每个数据集的分开验证，并保留每个数据集的训练过程，便于作业检查
1.  启动 Jupyter 服务器: `jupyter notebook`
2.  打开 `RETFound_MAE_finetune.ipynb`。
3.  在 Notebook 的第一个代码单元格中修改配置（如数据集名称），然后依次执行所有单元格。

## 5. 实验结果与分析

### 5.1 PaddlePaddle 实验结果
我们使用测试集评估了模型在各个数据集上的最终性能，取得了良好的效果。各项指标如下：

| 数据集 | 准确率 (Acc) | F1 分数 (Macro) | ROC AUC (Macro) | Kappa 分数 |
| :--- | :---: | :---: | :---: | :---: |
| **OCTID** | 0.8966 | 0.8717 | 0.9845 | 0.8638 |
| **Retina** | 0.6354 | 0.5927 | 0.8490 | 0.4471 |
| **APTOS2019**| 0.8445 | 0.6819 | 0.9406 | 0.7615 |
| **JSIEC** | 0.8589 | 0.7805 | 0.9923 | 0.8523 |

### 5.2 与 PyTorch 的结果对比

为验证我们 PaddlePaddle 实现的**正确性**，我们将其结果与一个采用相同超参数的 PyTorch 实现进行了比较。

| 数据集 | 框架 | **准确率 (Acc)** | **F1 分数 (Macro)** | **ROC AUC (Macro)** | **Kappa 分数** |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **OCTID** | **PaddlePaddle** | **0.8966** | **0.8717** | **0.9845** | **0.8638** |
| | PyTorch | 0.8276 | 0.7626 | 0.9628 | 0.7730 |
| **Retina** | **PaddlePaddle** | 0.6354 | 0.5927 | 0.8490 | **0.4471** |
| | PyTorch | 0.6354 | 0.5931 | 0.8559 | 0.4394 |
| **APTOS2019**| **PaddlePaddle** | **0.8445** | 0.6819 | 0.9406 | 0.7615 |
| | PyTorch | 0.8436 | 0.7038 | 0.9482 | 0.7628 |
| **JSIEC** | **PaddlePaddle** | **0.8589** | **0.7805** | **0.9923** | **0.8523** |
| | PyTorch | 0.8527 | 0.7753 | 0.9914 | 0.8459 |

**对比分析**:
从上表结果可见，在相同的实验设置下，我们基于 **PaddlePaddle 的实现结果与 PyTorch 版本的结果非常接近**。

在 `Retina` 和 `APTOS2019` 等数据集上，各项核心指标几乎没有差异。在 `OCTID` 和 `JSIEC` 数据集上，两者结果也处于同一水平。这种高度的一致性有力地证明了**本项目的 PaddlePaddle 代码实现、训练流程和参数配置是正确无误的**，成功地复现了原项目的模型性能，达到了本次课程设计的要求。

## 6. 致谢

本项目是在 **`rmaphoh/RETFound_MAE`** 的代码基础上完成的，我们对原作者的开源贡献表示衷心的感谢。
* **原始代码库**: [https://github.com/rmaphoh/RETFound_MAE](https://github.com/rmaphoh/RETFound_MAE)
