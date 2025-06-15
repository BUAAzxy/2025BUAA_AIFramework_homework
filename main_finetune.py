# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------
import warnings

warnings.filterwarnings("ignore")
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import paddle
import torch
from visualdl import LogWriter
import models_vit as models
import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from huggingface_hub import hf_hub_download
from engine_finetune import train_one_epoch, evaluate
import paddle.distributed as dist
import faulthandler

faulthandler.enable()


class Mixup:
    """
    实现Mixup和Cutmix数据增强的类。
    在本次实验未使用。
    """

    def __init__(
        self,
        mixup_alpha=1.0,
        cutmix_alpha=0.0,
        cutmix_minmax=None,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=0.1,
        num_classes=1000,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        self.mixup_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = mode
        self.mixup_enabled = mixup_alpha > 0.0 or cutmix_alpha > 0.0

    def _one_hot(self, x, num_classes, on_value=1.0, off_value=0.0):
        x = x.long().reshape([-1, 1])
        return paddle.full((x.shape[0], num_classes), off_value).scatter_(
            1, x, on_value
        )

    def __call__(self, x, target):
        if not self.mixup_enabled:
            return (x, target)
        use_cutmix = np.random.rand() < self.switch_prob
        if use_cutmix:
            lam, use_cutmix = self._params_for_cutmix()
        else:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        if self.mode == "batch":
            rand_index = paddle.randperm(x.shape[0])
            if use_cutmix:
                x_shuffled, y_shuffled = self._cutmix(x, target, rand_index, lam)
            else:
                x_shuffled = x[rand_index]
                x = x * lam + x_shuffled * (1.0 - lam)
                y_shuffled = target[rand_index]
            target = self._mix_labels(target, y_shuffled, lam)
        else:
            raise ValueError(f"Unsupported mixup mode: {self.mode}")
        return (x, target)

    def _params_for_cutmix(self):
        if self.cutmix_minmax is not None:
            lam = np.random.uniform(self.cutmix_minmax[0], self.cutmix_minmax[1])
        else:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        return (lam, True)

    def _cutmix(self, x, target, rand_index, lam):
        W, H = (x.shape[2], x.shape[3])
        r_x = np.random.randint(W)
        r_y = np.random.randint(H)
        r_w = int(W * np.sqrt(1 - lam))
        r_h = int(H * np.sqrt(1 - lam))
        x1 = np.clip(r_x - r_w // 2, 0, W)
        x2 = np.clip(r_x + r_w // 2, 0, W)
        y1 = np.clip(r_y - r_h // 2, 0, H)
        y2 = np.clip(r_y + r_h // 2, 0, H)
        x_shuffled = x[rand_index]
        x[:, :, y1:y2, x1:x2] = x_shuffled[:, :, y1:y2, x1:x2]
        lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
        return (x, target[rand_index])

    def _mix_labels(self, y1, y2, lam):
        if self.label_smoothing > 0:
            off_value = self.label_smoothing / self.num_classes
            on_value = 1.0 - self.label_smoothing + off_value
        else:
            on_value, off_value = (1.0, 0.0)
        y1 = self._one_hot(y1, self.num_classes, on_value, off_value)
        y2 = self._one_hot(y2, self.num_classes, on_value, off_value)
        return y1 * lam + y2 * (1.0 - lam)


def get_args_parser():
    """
    定义并解析所有命令行参数。
    这是整个微调任务的配置中心。

    Returns:
        argparse.ArgumentParser: 一个包含了所有已定义参数的解析器对象。
    """
    parser = argparse.ArgumentParser(
        "MAE fine-tuning for image classification", add_help=False
    )
    # --- 核心参数 ---
    parser.add_argument("--batch_size", default=16, type=int, help="每个GPU的批量大小")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--accum_iter", default=1, type=int, help="梯度累积步数")
    parser.add_argument(
        "--model",
        default="RETFound_mae",
        type=str,
        metavar="MODEL",
        help="要训练的模型名称",
    )
    parser.add_argument("--input_size", default=224, type=int, help="图像输入尺寸")
    parser.add_argument("--lr", type=float, default=None, metavar="LR", help="学习率")
    parser.add_argument(
        "--layer_decay", type=float, default=0.65, help="分层学习率衰减因子"
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=10, metavar="N", help="学习率预热的周期数"
    )
    parser.add_argument(
        "--finetune", default="RETFound_mae_meh", help="要加载的预训练模型检查点"
    )
    parser.add_argument("--data_path", default="./OCTID", type=str, help="数据集路径")
    parser.add_argument("--nb_classes", default=5, type=int, help="分类任务的类别数")
    parser.add_argument(
        "--task", default="RETFound_mae_meh-OCTID", type=str, help="微调任务的名称"
    )
    # --- 其他参数 ---
    parser.add_argument(
        "--drop_path", type=float, default=0.2, metavar="PCT", help="Drop path rate"
    )
    parser.add_argument(
        "--clip_grad", type=float, default=None, metavar="NORM", help="梯度裁剪"
    )
    parser.add_argument("--weight_decay", type=float, default=0.05, help="权重衰减")
    parser.add_argument(
        "--blr", type=float, default=0.005, metavar="LR", help="基础学习率"
    )
    parser.add_argument(
        "--min_lr", type=float, default=1e-06, metavar="LR", help="学习率下限"
    )
    parser.add_argument(
        "--color_jitter", type=float, default=None, metavar="PCT", help="颜色抖动"
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help="自动增强策略",
    )
    parser.add_argument("--smoothing", type=float, default=0.1, help="标签平滑")
    parser.add_argument(
        "--reprob", type=float, default=0.25, metavar="PCT", help="随机擦除概率"
    )
    parser.add_argument("--mixup", type=float, default=0, help="mixup alpha")
    parser.add_argument("--cutmix", type=float, default=0, help="cutmix alpha")
    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=True)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="使用CLS Token替代全局池化",
    )
    parser.add_argument("--output_dir", default="./output_dir", help="输出路径")
    parser.add_argument("--log_dir", default="./output_logs", help="日志路径")
    parser.add_argument("--device", default="gpu:0", help="使用的设备")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="从检查点恢复训练")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N")
    parser.add_argument("--eval", action="store_true", help="仅执行评估")
    parser.add_argument("--dist_eval", action="store_true", default=False)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--savemodel", action="store_true", default=True)
    parser.add_argument("--norm", default="IMAGENET", type=str)
    parser.add_argument("--enhance", action="store_true", default=False)
    parser.add_argument("--datasets_seed", default=2026, type=int)
    return parser


def main(args, criterion):
    """
    主执行函数，编排整个微调流程。

    Args:
        args: 解析后的命令行参数。
        criterion: 损失函数。
    """
    # 初始化环境和模型
    misc.init_distributed_mode(args)
    device = paddle.set_device(args.device)
    seed = args.seed + misc.get_rank()
    paddle.seed(seed)
    np.random.seed(seed)
    if args.model == "RETFound_mae":
        model = models.__dict__[args.model](
            img_size=args.input_size,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
    else:
        model = models.__dict__[args.model](num_classes=args.nb_classes)

    # 加载和适配预训练权重
    if args.finetune and (not args.eval):
        checkpoint_path = hf_hub_download(
            repo_id=f"YukunZhou/{args.finetune}", filename=f"{args.finetune}.pth"
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if args.model != "RETFound_mae":
            checkpoint_model = checkpoint["teacher"]
        else:
            checkpoint_model = checkpoint["model"]
        checkpoint_model = {
            k: v for k, v in checkpoint_model.items() if not k.startswith("decoder")
        }
        checkpoint_model = {
            k.replace("backbone.", ""): v for k, v in checkpoint_model.items()
        }
        checkpoint_model = {
            k.replace("mlp.w12.", "mlp.fc1."): v for k, v in checkpoint_model.items()
        }
        checkpoint_model = {
            k.replace("mlp.w3.", "mlp.fc2."): v for k, v in checkpoint_model.items()
        }
        paddle_state_dict = {}
        for k, v in checkpoint_model.items():
            if v.ndim == 2:
                v = v.t()  # 核心：转置线性层权重
            paddle_state_dict[k] = paddle.to_tensor(v.numpy())
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if (
                k in paddle_state_dict
                and paddle_state_dict[k].shape != state_dict[k].shape
            ):
                del paddle_state_dict[k]
        interpolate_pos_embed(model, paddle_state_dict)
        msg = model.set_state_dict(paddle_state_dict)
        feature_dim = model.embed_dim
        model.head = paddle.nn.Linear(feature_dim, args.nb_classes)
        trunc_normal_ = paddle.nn.initializer.TruncatedNormal(std=2e-05)
        trunc_normal_(model.head.weight)
        if hasattr(model.head, "bias") and model.head.bias is not None:
            paddle.nn.initializer.Constant(0.0)(model.head.bias)

    # 准备数据加载器
    dataset_train = build_dataset(is_train="train", args=args)
    dataset_val = build_dataset(is_train="val", args=args)
    dataset_test = build_dataset(is_train="test", args=args)
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    if not args.eval:
        sampler_train = paddle.io.DistributedBatchSampler(
            dataset=dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )
    if args.dist_eval:
        sampler_val = paddle.io.DistributedBatchSampler(
            dataset=dataset_val, batch_size=args.batch_size, shuffle=True
        )
    else:
        sampler_val = paddle.io.BatchSampler(
            dataset=dataset_val, batch_size=args.batch_size, shuffle=False
        )
        sampler_test = paddle.io.BatchSampler(
            dataset=dataset_test, batch_size=args.batch_size, shuffle=False
        )
    log_writer = None
    if global_rank == 0 and args.log_dir is not None and (not args.eval):
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = LogWriter(logdir=os.path.join(args.log_dir, args.task))
    if not args.eval:
        data_loader_train = paddle.io.DataLoader(
            dataset_train,
            batch_sampler=sampler_train,
            num_workers=args.num_workers,
            use_shared_memory=args.pin_mem,
        )
        data_loader_val = paddle.io.DataLoader(
            dataset_val,
            batch_sampler=sampler_val,
            num_workers=args.num_workers,
            use_shared_memory=args.pin_mem,
        )
    data_loader_test = paddle.io.DataLoader(
        dataset_test,
        batch_sampler=sampler_test,
        num_workers=args.num_workers,
        use_shared_memory=args.pin_mem,
    )

    # 设置优化器和损失函数
    model_without_ddp = model
    if num_tasks > 1:
        model = paddle.DataParallel(model)
        model_without_ddp = model._layers
    eff_batch_size = args.batch_size * args.accum_iter * num_tasks
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    param_groups = lrd.param_groups_lrd(
        model_without_ddp,
        args.weight_decay,
        no_weight_decay_list=[],
        layer_decay=args.layer_decay,
    )
    optimizer = paddle.optimizer.AdamW(parameters=param_groups, learning_rate=args.lr)
    loss_scaler = NativeScaler()
    misc.load_model_optimizer_scaler(args, model_without_ddp, optimizer, loss_scaler)

    # 开始训练和评估循环
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_score = 0.0
    best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if num_tasks > 1:
            data_loader_train.batch_sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            mixup_fn=None,
            log_writer=log_writer,
            args=args,
        )
        val_stats, val_score = evaluate(
            data_loader_val,
            model,
            device,
            args,
            epoch,
            mode="val",
            num_class=args.nb_classes,
            log_writer=log_writer,
        )
        if misc.is_main_process(): 
            # ── 拼接训练指标 ──
            train_msg = " | ".join([f"{k}={v:.4f}" for k, v in train_stats.items()])
            # ── 拼接验证指标 ──
            val_msg = " | ".join([f"{k}={v:.4f}" for k, v in val_stats.items()])
            print(
                f"Epoch {epoch+1:>3}/{args.epochs} │ "
                f"[train] {train_msg} │ "
                f"[val] {val_msg} │ "
                f"val_score={val_score:.4f}"
            )
        if val_score > max_score:
            max_score = val_score
            best_epoch = epoch
            if args.output_dir and args.savemodel:
                misc.save_model_paddle(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    mode="best",
                )
        if epoch == args.epochs - 1:
            best_model_path = os.path.join(
                args.output_dir, args.task, "checkpoint-best.pdparams"
            )
            if os.path.exists(best_model_path):
                checkpoint = paddle.load(best_model_path)
                model_without_ddp.set_state_dict(checkpoint)
                test_stats, _ = evaluate(
                    data_loader_test,
                    model,
                    device,
                    args,
                    epoch=best_epoch,
                    mode="test",
                    num_class=args.nb_classes,
                )

        if log_writer is not None:
            log_writer.add_scalar("loss/val", val_stats["loss"], epoch)
            log_writer.add_scalar("score/val", val_score, epoch)
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
            "epoch": epoch,
        }
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, args.task, "log.txt"),
                mode="a",
                encoding="utf-8",
            ) as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    print(f"Training time {str(datetime.timedelta(seconds=int(total_time)))}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    criterion = paddle.nn.CrossEntropyLoss()
    if args.output_dir:
        Path(os.path.join(args.output_dir, args.task)).mkdir(
            parents=True, exist_ok=True
        )
    main(args, criterion)
