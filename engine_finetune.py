# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
import csv
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Optional
from timm.data import Mixup
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, average_precision_score,
    hamming_loss, jaccard_score, recall_score, precision_score, cohen_kappa_score
)
from pycm import ConfusionMatrix
import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(
    model: nn.Layer,
    criterion: nn.Layer,
    data_loader: Iterable,
    optimizer: paddle.optimizer.Optimizer,
    device: str,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[callable] = None,
    log_writer=None,
    args=None,
):
    """
    在单个周期（epoch）上训练模型。
    
    Args:
        model (nn.Layer): 要训练的模型。
        criterion (nn.Layer): 损失函数。
        data_loader (Iterable): 训练数据加载器。
        optimizer (paddle.optimizer.Optimizer): 优化器。
        device (str): 训练设备。
        epoch (int): 当前周期数。
        loss_scaler: 用于混合精度训练的梯度缩放器。
        max_norm (float): 梯度裁剪的最大范数。
        mixup_fn (callable, optional): Mixup/Cutmix数据增强函数。
        log_writer: 用于写入日志的记录器 (e.g., VisualDL)。
        args: 包含所有超参数的命名空间。

    Returns:
        dict: 包含训练损失和学习率平均值的字典。
    """
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    print_freq, accum_iter = (20, args.accum_iter)
    optimizer.clear_grad()

    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, f"Epoch: [{epoch}]")
    ):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        current_place = paddle.get_device()
        samples = paddle.to_tensor(samples, place=current_place)
        targets = paddle.to_tensor(targets, place=current_place)

        if mixup_fn:
            samples, targets = mixup_fn(samples, targets)

        with paddle.amp.auto_cast(enable=True, level="O1"):
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()
        loss = loss / accum_iter

        loss_scaler(
            loss, optimizer, clip_grad=max_norm, parameters=model.parameters(),
            create_graph=False, update_grad=(data_iter_step + 1) % accum_iter == 0,
        )

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.clear_grad()
        
        if "gpu" in current_place:
            device_index = int(current_place.split(":")[1])
            paddle.device.cuda.synchronize(device=device_index)

        metric_logger.update(loss=loss_value, lr=optimizer.get_lr())
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("loss/train", loss_value_reduce, step=epoch_1000x)
            log_writer.add_scalar("lr", optimizer.get_lr(), step=epoch_1000x)

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@paddle.no_grad()
def evaluate(
    data_loader, model: nn.Layer, device: str, args, epoch, mode, num_class, log_writer=None,
):
    """
    在验证集或测试集上评估模型性能。
    
    Args:
        data_loader: 评估数据加载器。
        model (nn.Layer): 要评估的模型。
        device (str): 评估设备。
        args: 包含所有超参数的命名空间。
        epoch (int): 当前周期数，用于日志记录。
        mode (str): 评估模式 ('val' 或 'test')。
        num_class (int): 分类任务的类别数。
        log_writer: 用于写入日志的记录器。

    Returns:
        tuple: 一个包含评估指标字典和综合得分（score）的元组。
    """
    criterion = nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)
    model.eval()

    true_onehot, pred_onehot, true_labels, pred_labels, pred_softmax = ([], [], [], [], [])
    current_place = paddle.get_device()

    for batch in metric_logger.log_every(data_loader, 10, f"{mode}:"):
        images = paddle.to_tensor(batch[0], place=current_place)
        target = paddle.to_tensor(batch[-1], place=current_place)
        target_onehot = F.one_hot(target.astype("int64"), num_classes=num_class)

        with paddle.amp.auto_cast(enable=True, level="O1"):
            output = model(images)
            loss = criterion(output, target)

        output_ = F.softmax(output, axis=1)
        output_label = paddle.argmax(output_, axis=1)
        output_onehot = F.one_hot(output_label.astype("int64"), num_classes=num_class)
        
        metric_logger.update(loss=loss.item())
        
        true_onehot.extend(target_onehot.numpy())
        pred_onehot.extend(output_onehot.numpy())
        true_labels.extend(target.numpy())
        pred_labels.extend(output_label.numpy())
        pred_softmax.extend(output_.numpy())

    accuracy = accuracy_score(true_labels, pred_labels)
    hamming = hamming_loss(true_onehot, pred_onehot)
    jaccard = jaccard_score(true_onehot, pred_onehot, average="macro")
    average_precision = average_precision_score(true_onehot, pred_softmax, average="macro")
    kappa = cohen_kappa_score(true_labels, pred_labels)
    f1 = f1_score(true_onehot, pred_onehot, zero_division=0, average="macro")
    roc_auc = roc_auc_score(true_onehot, pred_softmax, multi_class="ovr", average="macro")
    precision = precision_score(true_onehot, pred_onehot, zero_division=0, average="macro")
    recall = recall_score(true_onehot, pred_onehot, zero_division=0, average="macro")

    score = (f1 + roc_auc + kappa) / 3

    if log_writer:
        for metric_name, value in zip(
            ["accuracy", "f1", "roc_auc", "hamming", "jaccard", "precision", "recall", "average_precision", "kappa", "score"],
            [accuracy, f1, roc_auc, hamming, jaccard, precision, recall, average_precision, kappa, score],
        ):
            log_writer.add_scalar(tag=f"perf/{metric_name}", value=value, step=epoch)

    metric_logger.synchronize_between_processes()
    
    results_path = os.path.join(args.output_dir, args.task, f"metrics_{mode}.csv")
    file_exists = os.path.isfile(results_path)
    with open(results_path, "a", newline="", encoding="utf8") as cfa:
        wf = csv.writer(cfa)
        if not file_exists:
            wf.writerow(["val_loss", "accuracy", "f1", "roc_auc", "hamming", "jaccard", "precision", "recall", "average_precision", "kappa"])
        wf.writerow([metric_logger.meters["loss"].global_avg, accuracy, f1, roc_auc, hamming, jaccard, precision, recall, average_precision, kappa])

    if mode == "test":
        cm = ConfusionMatrix(actual_vector=true_labels, predict_vector=pred_labels)
        cm.plot(cmap=plt.cm.Blues, number_label=True, normalized=True, plot_lib="matplotlib")
        plt.savefig(
            os.path.join(args.output_dir, args.task, "confusion_matrix_test.jpg"),
            dpi=600,
            bbox_inches="tight",
        )
    return ({k: meter.global_avg for k, meter in metric_logger.meters.items()}, score)