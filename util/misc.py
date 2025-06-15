# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import argparse
import builtins
import datetime
import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path
import paddle
import paddle.distributed as dist
from math import inf
import torch


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = paddle.to_tensor(
            [self.count, self.total, self.total_square], dtype="float64"
        )
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]
        self.total_square = t[2]

    @property
    def median(self):
        if not self.deque:
            return 0.0
        d = paddle.to_tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        if not self.deque:
            return 0.0
        d = paddle.to_tensor(list(self.deque), dtype=paddle.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        if self.count == 0:
            return 0.0
        return self.total / self.count

    @property
    def max(self):
        if not self.deque:
            return 0.0
        return max(self.deque)

    @property
    def value(self):
        if not self.deque:
            return 0.0
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, paddle.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg_parts = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if paddle.is_compiled_with_cuda() and "gpu" in paddle.get_device():
            log_msg_parts.append("max mem: {memory:.0f}MB")
        log_msg = self.delimiter.join(log_msg_parts)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                log_kwargs = {
                    "i": i,
                    "len_iterable": len(iterable),
                    "eta": eta_string,
                    "meters": str(self),
                    "time": str(iter_time),
                    "data": str(data_time),
                }
                if paddle.is_compiled_with_cuda() and "gpu" in paddle.get_device():
                    log_kwargs["memory"] = (
                        paddle.device.cuda.max_memory_allocated() / MB
                    )
                    current_log_msg = log_msg.format(
                        i,
                        len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time),
                        memory=paddle.device.cuda.max_memory_allocated() / MB,
                    )
                else:
                    current_log_msg = log_msg.format(
                        i,
                        len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time),
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print_with_timestamp(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            now = datetime.datetime.now().time().strftime("%H:%M:%S.%f")[:-3]
            builtin_print(f"[{now}] ", end="")
            builtin_print(*args, **kwargs)

    builtins.print = print_with_timestamp


def is_dist_avail_and_initialized():
    """检查分布式环境是否可用且已初始化"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        paddle.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        args.local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        args.device = f"gpu:{args.local_rank}"
        args.dist_url = "tcp://%s:%s" % (
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
        )
        os.environ["LOCAL_RANK"] = str(args.local_rank)
        os.environ["RANK"] = str(args.rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.device = f"gpu:{args.local_rank}"
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.local_rank = args.rank % paddle.device.cuda.device_count()
        args.device = f"gpu:{args.local_rank}"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
        args.dist_url = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    else:
        args.distributed = False
        if get_rank() == 0:
            setup_for_distributed(is_master=True)
        return
    args.distributed = True
    dist.init_parallel_env()
    dist.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = paddle.amp.GradScaler(init_loss_scaling=2.0**16)

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        if self._scaler is not None:
            scaled_loss = self._scaler.scale(loss)
            scaled_loss.backward()
            if update_grad:
                if clip_grad is not None:
                    assert parameters is not None
                    self._scaler.unscale_(optimizer)
                    norm = paddle.nn.utils.clip_grad_norm_(parameters, clip_grad)
                else:
                    self._scaler.unscale_(optimizer)
                self._scaler.step(optimizer)
                self._scaler.update()
                self.grad_norm = norm if clip_grad is not None else None
        else:
            loss.backward()
            if update_grad:
                if clip_grad is not None:
                    assert parameters is not None
                    norm = paddle.nn.utils.clip_grad_norm_(parameters, clip_grad)
                else:
                    norm = None
                optimizer.step()
                self.grad_norm = norm
        return self.grad_norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0):
    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return paddle.to_tensor(0.0)
    norm_type = float(norm_type)
    all_grads = [p.grad.detach() for p in parameters]
    if norm_type == inf:
        total_norm = max((g.abs().max() for g in all_grads))
    else:
        norms = [paddle.linalg.norm(g, p=norm_type) for g in all_grads]
        if not norms:
            return paddle.to_tensor(0.0)
        total_norm = paddle.linalg.norm(paddle.stack(norms), p=norm_type)
    return total_norm


def save_model_paddle(
    args, epoch, model, model_without_ddp, optimizer, loss_scaler, mode
):
    """PaddlePaddle specific model saving"""
    output_path = Path(args.output_dir) / args.task
    output_path.mkdir(parents=True, exist_ok=True)
    if mode == "best":
        save_filename = "checkpoint-best.pdparams"
        epoch_filename = "checkpoint-best.epoch.json"
    else:
        save_filename = "checkpoint-latest.pdparams"
        epoch_filename = "checkpoint-latest.epoch.json"
    checkpoint_path = output_path / save_filename
    epoch_info_path = output_path / epoch_filename
    to_save_model = model_without_ddp.state_dict()
    save_on_master(to_save_model, str(checkpoint_path))
    epoch_info = {
        "epoch": epoch,
        "args": vars(args) if isinstance(args, argparse.Namespace) else args,
    }
    if is_main_process():
        with open(epoch_info_path, "w") as f:
            json.dump(epoch_info, f, indent=2)
    if mode == "latest" and epoch < args.epochs - 1:
        opt_scaler_save_filename = (
            f"checkpoint-latest-opt-scaler-epoch-{epoch}.pdstates"
        )
        opt_scaler_path = output_path / opt_scaler_save_filename
        to_save_opt_scaler = {
            "optimizer": optimizer.state_dict(),
            "scaler": loss_scaler.state_dict() if loss_scaler else None,
            "epoch": epoch,
        }
        save_on_master(to_save_opt_scaler, str(opt_scaler_path))


def load_model_optimizer_scaler(args, model_without_ddp, optimizer, loss_scaler):
    """PaddlePaddle specific loading of model, optimizer, and scaler"""
    if args.resume:
        if args.resume.endswith(".pdparams"):
            model_checkpoint_path = args.resume
        else:
            model_checkpoint_path = args.resume
        if os.path.exists(model_checkpoint_path):
            state_dict = paddle.load(model_checkpoint_path)
            if "model" in state_dict and isinstance(state_dict, dict):
                model_without_ddp.set_state_dict(state_dict["model"])
            else:
                model_without_ddp.set_state_dict(state_dict)
            epoch_file_path = model_checkpoint_path.replace(".pdparams", ".epoch.json")
            if os.path.exists(epoch_file_path):
                with open(epoch_file_path, "r") as f:
                    epoch_info = json.load(f)
                    args.start_epoch = epoch_info.get("epoch", args.start_epoch) + 1
            if not args.eval and args.start_epoch > 0:
                opt_scaler_checkpoint_path = os.path.join(
                    Path(model_checkpoint_path).parent,
                    f"checkpoint-latest-opt-scaler-epoch-{args.start_epoch - 1}.pdstates",
                )
                if os.path.exists(opt_scaler_checkpoint_path):
                    opt_scaler_states = paddle.load(opt_scaler_checkpoint_path)
                    if "optimizer" in opt_scaler_states and optimizer:
                        optimizer.set_state_dict(opt_scaler_states["optimizer"])
                    if "scaler" in opt_scaler_states and loss_scaler:
                        loss_scaler.load_state_dict(opt_scaler_states["scaler"])


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        current_device = paddle.get_device()
        if isinstance(x, (int, float)):
            x_tensor = paddle.to_tensor(
                x,
                place=current_device if "gpu" in current_device else "cpu",
                dtype="float64",
            )
        elif isinstance(x, paddle.Tensor):
            x_tensor = x.astype("float64")
            if str(x.place) != current_device:
                x_tensor = paddle.to_tensor(
                    x_tensor, place=current_device if "gpu" in current_device else "cpu"
                )
        else:
            raise TypeError(f"Unsupported type for all_reduce_mean: {type(x)}")
        dist.all_reduce(x_tensor)
        return (x_tensor / world_size).item()
    else:
        return x


def convert_pytorch_to_paddle_state_dict(pytorch_state_dict, model_name_for_keys=""):
    """
    Converts a PyTorch state_dict to PaddlePaddle format.
    Includes key name replacements and tensor format conversion.
    """
    paddle_state_dict = {}
    temp_renamed_state_dict = {}
    for k, v in pytorch_state_dict.items():
        new_k = k
        new_k = new_k.replace("backbone.", "")
        if "RETFound_mae" in model_name_for_keys:
            new_k = new_k.replace("mlp.w12.", "mlp.fc1.")
            new_k = new_k.replace("mlp.w3.", "mlp.fc2.")
        temp_renamed_state_dict[new_k] = v
    for k, v_torch in temp_renamed_state_dict.items():
        if isinstance(v_torch, torch.Tensor):
            numpy_array = v_torch.cpu().numpy()
            if "weight" in k and len(numpy_array.shape) == 2:
                is_linear_layer_weight = any(
                    (
                        substr in k
                        for substr in [
                            "fc",
                            "qkv",
                            "proj",
                            "head",
                            "attention. Aufmerksamkeit.",
                            "out_proj",
                        ]
                    )
                )
                is_embedding = "embed" in k
                if is_linear_layer_weight and (not is_embedding):
                    numpy_array = numpy_array.T
            paddle_state_dict[k] = paddle.to_tensor(numpy_array)
    return paddle_state_dict


def get_appropriate_resume_path(resume_path_arg):
    """
    Determines the actual checkpoint file to load based on the resume argument.
    This could involve checking for specific file extensions or existence.
    """
    if os.path.isfile(resume_path_arg):
        return resume_path_arg
    if os.path.exists(resume_path_arg + ".pdparams"):
        return resume_path_arg + ".pdparams"
    if os.path.exists(resume_path_arg + ".pth"):
        return resume_path_arg + ".pth"
    return resume_path_arg
