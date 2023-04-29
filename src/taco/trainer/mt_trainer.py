import logging
import os
import sys
from itertools import repeat
from typing import Any, Dict, List, Optional, Tuple, Union
import pickle
import numpy as np
import torch.nn.functional as F
from scipy.stats import entropy
import random
import time
import datasets
import torch
import torch.nn as nn
from packaging import version
from collections import defaultdict
from copy import deepcopy
from torch.nn.parallel.distributed import _find_tensors
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from ..utils import get_task_hps
from transformers.trainer import Trainer, TRAINER_STATE_NAME
from transformers.trainer_pt_utils import IterableDatasetShard, \
    is_sagemaker_mp_enabled, nested_detach

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from transformers.trainer_pt_utils import smp_forward_backward, \
        smp_forward_only, smp_nested_concat
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from transformers.trainer_pt_utils import DistributedSamplerWithLoop
from ..dataset import QPCollator
from ..utils import MultiTaskDataLoader
from transformers.integrations import TensorBoardCallback, rewrite_logs, \
    hp_params
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.trainer_utils import (
    HPSearchBackend,
    ShardedDDPOption,
    TrainOutput,
    has_length,
    speed_metrics
)
from transformers.deepspeed import deepspeed_init
from transformers.trainer_callback import TrainerState
from transformers.utils import (
    is_apex_available,
    is_datasets_available,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
)

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl
if is_apex_available():
    from apex import amp
from ..loss import DistributedContrastiveLoss, SimpleContrastiveLoss
import math
from .dense_trainer import DenseTrainer

logger = logging.getLogger(__name__)

try:
    from grad_cache import GradCache

    _grad_cache_available = True
except ModuleNotFoundError:
    _grad_cache_available = False

"""
support features:

1. single multi_task trainer with various algorithms: naive, pcg, cgd, 
   gradnorm, taco
2. test deepspeed compatibility

"""


# support option: log grad norm or not, figure out if nan conditions
#  can be modified without relying on grad norm
# taco warmup steps calculated in init
# TODO: if hard negative mining style training, can we only warmup for the
#  first epoch for taco?


class MTDenseTrainer(DenseTrainer):

    def __init__(self, data_args=None, *args, **kwargs):
        super(MTDenseTrainer, self).__init__(*args, **kwargs)
        self.data_args = data_args
        self.task_names = self.args.task_names.split(',')
        self.num_tasks = len(self.task_names)
        self.n_passages = get_task_hps(data_args.mt_train_n_passages,
                                       self.num_tasks)
        self.q_max_lens = get_task_hps(data_args.q_max_lens, self.num_tasks)
        self.p_max_lens = get_task_hps(data_args.p_max_lens, self.num_tasks)
        self.task_batch_sizes, self.per_device_task_batch_sizes = \
            self.get_task_batch_sizes()
        self.data_collators = self.get_task_collators()
        self.up_sample = data_args.up_sample
        assert self.num_tasks == len(self.task_batch_sizes)
        if self.args.weight_method == 'cgd':
            ws = torch.ones(self.num_tasks) / self.num_tasks
            self.ws = ws.to(self.args.device)
        elif self.args.weight_method == 'gn':  # gradnorm
            self.ws = nn.Parameter(torch.ones(self.num_tasks).to(
                self.args.device))
            # self.num_steps = 0
            self.init_losses = torch.ones(self.num_tasks).to(self.args.device)
            # TODO: add train loss at time 0
        self.grad_idx_cumsum = self.get_grad_cumsum().tolist()
        self.grad_dim = self.grad_idx_cumsum[-1]
        if self.args.weight_method == 'taco':
            self.ipt_exp = torch.zeros((self.num_tasks, self.grad_dim)).to(
                self.args.device)
            self.eps = 1e-12
            logger.info(f'state max steps {self.state.max_steps}')
            logger.info(f'warmup ratio {self.args.warmup_ratio}')
            self.taco_warmup_steps = int(self.state.max_steps *
                                         self.args.warmup_ratio)
            logger.info(f'taco warmup steps {self.taco_warmup_steps}')
        if not self.args.select_all:
            self.selection_masks = self.get_selection_masks()
        if self.args.weight_method != 'naive':
            assert self.args.gradient_accumulation_steps == 1, \
                f"we don't allow gradient accumulation > 1 for " \
                f"{self.args.weight_method}"

    def get_selection_masks(self):
        masks = torch.zeros(self.grad_dim)
        if self.args.select_blocks is None:
            select = ['block.9', 'block.10', 'block.11',
                      'decoder.final_layer_norm']
        else:
            select = self.args.select_blocks.split(',')
        for i, (name, param) in enumerate(self.model.named_parameters()):
            for s in select:
                if name.find(s) >= 0:
                    beg = self.grad_idx_cumsum[i - 1] if i != 0 else 0
                    end = self.grad_idx_cumsum[i]
                    masks[beg:end] = 1
                    break
        return masks.bool()

    def unscale_grads(self, grads):
        inv_scale = 1. / self.scaler.get_scale()
        grads *= inv_scale
        return grads

    def scale_grads(self, grads):
        scale = self.scaler.get_scale()
        return grads * scale

    def get_grad_cumsum(self):
        grad_index = []
        for name, param in self.model.named_parameters():
            grad_index.append(param.data.numel())
        return np.cumsum(grad_index)

    def param2vec(self):
        ps = torch.zeros(self.grad_dim).to(self.args.device)
        for i, (name, param) in enumerate(self.model.named_parameters()):
            if param.data is not None:
                beg = 0 if i == 0 else self.grad_idx_cumsum[i - 1]
                end = self.grad_idx_cumsum[i]
                ps[beg:end] = param.data.view(-1)
        return ps

    def grad2vec(self):
        grad = torch.zeros(self.grad_dim)
        for i, (name, param) in enumerate(self.model.named_parameters()):
            if param.grad is not None:
                beg = 0 if i == 0 else self.grad_idx_cumsum[i - 1]
                end = self.grad_idx_cumsum[i]
                grad[beg:end] = param.grad.data.view(-1)
        return grad

    def get_grads(self, losses, model):
        # T x p
        grads = torch.zeros(self.num_tasks, self.grad_dim).to(self.args.device)
        for t in range(self.num_tasks):
            if self.args.world_size > 1:
                out_tensors = list(_find_tensors(losses))
                model.reducer.prepare_for_backward(out_tensors)
            if (t + 1) != self.num_tasks:
                if self.do_grad_scaling:
                    self.scaler.scale(losses[t]).backward(retain_graph=True)
                elif self.deepspeed:
                    # loss gets scaled under gradient_accumulation_steps in deepspeed
                    losses[t] = self.deepspeed.backward(losses[t],
                                                        retain_graph=True)
                else:
                    losses[t].backward(retain_graph=True)
            else:
                if self.do_grad_scaling:
                    self.scaler.scale(losses[t]).backward()
                elif self.deepspeed:
                    # loss gets scaled under gradient_accumulation_steps in deepspeed
                    losses[t] = self.deepspeed.backward(losses[t])
                else:
                    losses[t].backward()
            # self.clip_select_grads(max_grad_norm)
            grads[t] = self.grad2vec()
            # can't do zero_grad because if gradient accumulation
            model.zero_grad()
        return grads

    def reset_grads(self, new_grads):
        for i, (name, param) in enumerate(self.model.named_parameters()):
            if param.grad is not None:
                beg = 0 if i == 0 else self.grad_idx_cumsum[i - 1]
                end = self.grad_idx_cumsum[i]
                param.grad.data = new_grads[beg:end].contiguous().view(
                    param.data.size()).data.clone()

    def get_task_collators(self):
        collators = [QPCollator(
            self.tokenizer,
            max_q_len=q_max_len,
            max_p_len=p_max_len,
            multi_label=self.args.multi_label
        ) for q_max_len, p_max_len in zip(self.q_max_lens, self.p_max_lens)]
        return collators

    def get_task_batch_sizes(self):
        if self.args.per_device_train_batch_size < self.num_tasks:
            raise ValueError(
                "per device batch size should larger than number of tasks")
        len_list = [len(task_set) for task_set in self.train_dataset]
        mix_rates = [l / sum(len_list) for l in len_list]
        if self.args.multi_mix_temp != 1.0:
            mix_rates = [math.pow(r, 1 / self.args.multi_mix_temp) for r
                         in mix_rates]
            mix_rates = [r / sum(mix_rates) for r in mix_rates]

        per_device_task_batch_sizes = self._get_int_mixture(
            self.args.per_device_train_batch_size, mix_rates)
        task_batch_sizes = [b * max(1, self.args.n_gpu) for b in
                            per_device_task_batch_sizes]

        return task_batch_sizes, per_device_task_batch_sizes

    def _get_int_mixture(self, orig_bsz, mix_rates):
        # each task batch size is at least 1
        res_bsz = orig_bsz - self.num_tasks
        mr = [(orig_bsz * r - 1) / res_bsz for r in mix_rates]
        raw_mix = [r * res_bsz for r in mr]
        task_batch_sizes = [int(m) for m in raw_mix]
        while sum(task_batch_sizes) < res_bsz:
            res = [p - q for p, q in zip(raw_mix, task_batch_sizes)]
            task_batch_sizes[np.argmax(res)] += 1
        task_batch_sizes = [t + 1 for t in task_batch_sizes]
        return task_batch_sizes

    def get_single_loader(self, train_dataset, data_idx):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if is_datasets_available() and isinstance(train_dataset,
                                                  datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset,
                                                        description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.task_batch_sizes[data_idx],
                    drop_last=False,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.per_device_task_batch_sizes[data_idx],
                collate_fn=self.data_collators[data_idx],
                drop_last=False,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        if self.args.world_size <= 1:
            train_sampler = RandomSampler(train_dataset)
        elif (
                self.args.parallel_mode in ["tpu",
                                            "sagemaker_model_parallel"]
                and not self.args.dataloader_drop_last
        ):
            # Use a loop for TPUs when drop_last is False to have all batches have the same size.
            train_sampler = DistributedSamplerWithLoop(
                train_dataset,
                batch_size=self.per_device_task_batch_sizes[data_idx],
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                seed=self.args.seed,
            )
        else:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                seed=self.args.seed,
            )

        return DataLoader(
            train_dataset,
            batch_size=self.task_batch_sizes[data_idx],
            sampler=train_sampler,
            collate_fn=self.data_collators[data_idx],
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_train_dataloader(self):
        return MultiTaskDataLoader([self.get_single_loader(task_dataset, i) for
                                    i, task_dataset in
                                    enumerate(self.train_dataset)],
                                   self.up_sample
                                   )

    def get_eval_dataloader(self, eval_dataset: Optional = None):
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return MultiTaskDataLoader([self.get_single_loader(task_dataset, i) for
                                    i, task_dataset in
                                    enumerate(eval_dataset)],
                                   self.up_sample)

    def compute_loss(self, model, inputs, return_outputs=False):
        # each task has query, passage input
        losses = []
        if return_outputs:
            outputs = []
        assert len(inputs) == self.num_tasks
        for i, task_inputs in enumerate(inputs):
            if self.args.multi_label:
                query, passage, target = task_inputs
                single_outputs = model(query=query, passage=passage,
                                       target=target,
                                       n_passages=self.n_passages[i],
                                       task_name=self.task_names[i])
            else:
                query, passage = task_inputs
                single_outputs = model(query=query, passage=passage,
                                       n_passages=self.n_passages[i],
                                       task_name=self.task_names[i])
            losses.append(single_outputs.loss)
            if return_outputs:
                outputs.append(single_outputs)
        if return_outputs:
            outputs = [losses] + outputs
        return (losses, outputs) if return_outputs else losses

    def taco_backward(self, losses, model):
        losses = torch.stack(losses)
        # T x p
        grads = self.get_grads(losses,
                               model)
        if self.do_grad_scaling:
            grads = self.unscale_grads(grads)
        # grads_norm = grads.norm(dim=-1)
        # total_norm = grads_norm.sum()
        grads_norm = None
        if self.args.log_gnorm:
            grads_norm = grads.norm(dim=-1)
        if not torch.isfinite(grads).all():
            # if torch.logical_or(total_norm.isnan(), total_norm.isinf()):
            logger.info("Detect nan or inf grad! Use naive update ")
            grads = grads.sum(0)
            if self.do_grad_scaling:
                grads = self.scale_grads(grads)
            self.reset_grads(grads / self.num_tasks)
            loss = sum(losses) / self.num_tasks
        else:
            logger.info(f'warmup steps {self.taco_warmup_steps}')
            with torch.no_grad():
                # T x p
                p_datas = self.param2vec()
                ipt = (grads * p_datas).abs()
                if self.args.norm_ipt:
                    ipt = ipt / (ipt.median(dim=-1, keepdim=True)[0] + self.eps)
                self.ipt_exp.mul_(
                    self.args.beta_taco).add_(ipt,
                                              alpha=1.0 - self.args.beta_taco)
                if self.state.global_step < self.taco_warmup_steps:
                    grads = grads.sum(0)
                else:
                    if self.args.norm_ipt:
                        w_scores = self.ipt_exp / self.args.tau_taco
                    else:
                        w_scores = self.ipt_exp / (self.ipt_exp.median(
                            dim=-1, keepdim=True)[0] + self.eps)
                        w_scores /= self.args.tau_taco
                    if self.args.discourage:
                        w_scores *= -1
                    gw = F.softmax(w_scores, dim=0)
                    grads = (gw * grads).sum(0)
            if self.do_grad_scaling:
                grads = self.scale_grads(grads)
            self.reset_grads(grads)
            loss = sum(losses)
        return loss, grads_norm

    def pcg_backward(self, losses, model):
        grads = self.get_grads(losses, model)
        if self.do_grad_scaling:
            grads = self.unscale_grads(grads)
        # grads_norm = grads.norm(dim=-1)
        # total_norm = grads_norm.sum()
        grads_norm = None
        if not torch.isfinite(grads).all():
            logger.info("Detect nan or inf grad! Use naive update ")
            new_grads = grads.sum(0)
            if self.do_grad_scaling:
                new_grads = self.scale_grads(new_grads)
            self.reset_grads(new_grads / self.num_tasks)
            if self.args.log_gnorm:
                grads_norm = grads.norm(dim=-1)
        else:
            pc_grads = grads.clone()
            num_conflicts = 0
            for ti in range(self.num_tasks):
                task_index = list(range(self.num_tasks))
                random.shuffle(task_index)
                for tj in task_index:
                    gij = torch.dot(pc_grads[ti], grads[tj])
                    if gij < 0:
                        num_conflicts += 1
                        pc_grads[ti] -= gij * grads[tj] / (
                                grads[tj].norm() ** 2)
            new_grads = pc_grads.sum(0)
            if self.args.log_gnorm:
                grads_norm = new_grads.norm(dim=-1)
            # logger.info(f'num conflicts {num_conflicts}')
            if self.do_grad_scaling:
                new_grads = self.scale_grads(new_grads)
            self.reset_grads(new_grads / self.num_tasks)
            # loss = torch.matmul(self.ws, torch.stack(losses))
        loss = sum(losses) / self.num_tasks
        return loss, grads_norm

    def cgd_backward(self, losses, model):
        losses = torch.stack(losses)
        # T x p
        grads = self.get_grads(losses,
                               model)
        if self.do_grad_scaling:
            grads = self.unscale_grads(grads)
        # grads_norm = grads.norm(dim=-1, keepdim=True)
        # total_norm = grads_norm.sum()
        grads_norm = None
        if not torch.isfinite(grads).all():
            logger.info("Detect nan or inf grad! Use naive update ")
            new_grads = grads.sum(0)
            if self.do_grad_scaling:
                new_grads = self.scale_grads(new_grads)
            self.reset_grads(new_grads / self.num_tasks)
            loss = sum(losses) / self.num_tasks
        else:
            if self.args.norm_grad:
                grads_norm = grads.norm(dim=-1, keepdim=True)
                normed_grads = grads / grads_norm
            else:
                normed_grads = grads
            select_norm_grads = normed_grads if self.args.select_all else \
                normed_grads[:, self.selection_masks]
            cgd_grads = select_norm_grads.clone()
            with torch.no_grad():
                # T x p
                self.model.zero_grad()
                # T x T
                gg = torch.matmul(cgd_grads, cgd_grads.T)
                # T
                # (T x 1)  x  (1 x T) -> T x T
                lw = torch.matmul(losses.detach().unsqueeze(1),
                                  losses.detach().unsqueeze(0)).pow(
                    self.args.beta_cgd)
                ri = (lw * gg).sum(-1)
                self.ws = F.softmax((ri / self.args.tau_cgd) + self.ws.log(),
                                    dim=-1)
            new_grads = torch.matmul(self.ws, grads)
            if self.do_grad_scaling:
                new_grads = self.scale_grads(new_grads)
            self.reset_grads(new_grads)
            loss = torch.matmul(self.ws, losses)
        if self.args.log_gnorm and grads_norm is None:
            grads_norm = grads.norm(dim=-1)
        elif grads_norm is not None:
            grads_norm = grads_norm.view(-1)
        return loss, grads_norm

    def gn_backward(self, losses, model):
        # 1. get grads_t for tasks T x p
        # 2. get r_t = (loss_t/loss_t(0) / avg(loss_t/loss_t(0))
        # 3. grads_avg = avg_t(grads)
        # 4. const = grads_avg * r_t^alpha
        # 5. L_grad = sum_t |grads_i - const|
        # 6. backward again to compute gradients of w_i
        # T x p
        ws = self.num_tasks * F.softmax(self.ws, dim=-1)
        grads = self.get_grads(losses, model)
        if self.do_grad_scaling:
            grads = self.unscale_grads(grads)
        # grads_norm = grads.norm(dim=-1)
        # total_norm = grads_norm.sum()
        grads_norm = None
        if self.args.log_gnorm:
            grads_norm = grads.norm(dim=-1)
        stack_losses = torch.stack(losses)
        if self.state.global_step == 0:
            # get loss(0)
            self.init_losses = self._nested_gather(stack_losses[None]).mean(
                0).detach()
        G_per_loss = torch.norm(torch.matmul(ws, grads.detach()), p=2, dim=-1)
        G = G_per_loss.mean(0)
        if not torch.isfinite(grads).all():
            logger.info("Detect nan or inf grad! Use naive update ")
            new_grads = grads.sum(0)
            if self.do_grad_scaling:
                new_grads = self.scale_grads(new_grads)
            self.reset_grads(new_grads)
            loss = sum(losses)
        else:
            # p
            with torch.no_grad():
                Li = stack_losses / self.init_losses
                # p
                ri = Li / Li.mean()
                constant = (G * (ri ** self.args.beta_gn)).detach()
            L_grad = (G_per_loss - constant).abs().sum(0)
            L_grad.backward()
            loss_weight = ws.detach().clone()
            new_grads = torch.matmul(loss_weight, grads)
            if self.do_grad_scaling:
                new_grads = self.scale_grads(new_grads)
            self.reset_grads(new_grads)
            loss = torch.matmul(loss_weight, stack_losses)
        # self.num_steps += 1
        return loss, grads_norm

    def uw_backward(self, losses, model):
        # loss = sum(losses)
        loss = torch.matmul(self.ws * self.num_tasks, torch.stack(losses))
        grads = self.get_grads(losses, model)
        grads_norm = None
        if self.args.log_gnorm:
            grads_norm = grads.norm(dim=-1)
        if self.do_grad_scaling:
            grads = self.unscale_grads(grads)
        new_grads = grads.clone().sum(0)
        if self.do_grad_scaling:
            new_grads = self.scale_grads(new_grads)
        self.reset_grads(new_grads)
        return loss, grads_norm

    def training_step(self, model: torch.nn.Module,
                      inputs: Dict[
                          str, Union[torch.Tensor, Any]]):
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            losses = self.compute_loss(model, inputs)
        assert len(losses) == self.num_tasks

        if self.args.n_gpu > 1:
            for t in range(self.num_tasks):
                losses[t] = losses[
                    t].mean()  # mean() to average on multi-gpu parallel training
        if self.args.weight_method == 'naive':
            loss = sum(losses)
            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                loss = loss / self.args.gradient_accumulation_steps
            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.deepspeed:
                loss = self.deepspeed.backward(loss)
            else:
                loss.backward()
            grads_norm = None
        elif self.args.weight_method == 'pcg':
            # loss = self.naive_backward(losses, model)
            loss, grads_norm = self.pcg_backward(losses, model)
        elif self.args.weight_method == 'cgd':
            loss, grads_norm = self.cgd_backward(losses, model)
            # loss = self.naive_backward(losses, model)
        elif self.args.weight_method == 'uw':
            loss, grads_norm = self.uw_backward(losses, model)
        elif self.args.weight_method == 'taco':
            loss, grads_norm = self.taco_backward(losses, model)
        elif self.args.weight_method == 'gn':
            loss, grads_norm = self.gn_backward(losses, model)
        else:
            raise NotImplementedError
        # return loss.detach() / self._dist_loss_scale_factor
        task_losses = [l.detach() / self._dist_loss_scale_factor for l in
                       losses]
        task_losses = torch.tensor(task_losses).to(self.args.device)
        return loss.detach() / self._dist_loss_scale_factor, task_losses, \
               grads_norm

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        # has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        # prediction_loss_only = True

        with torch.no_grad():
            with self.compute_loss_context_manager():
                losses, outputs = self.compute_loss(model, inputs,
                                                    return_outputs=True)
            loss = sum(losses)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: Dict[str, Any]) -> None:
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state,
                                                    self.control, logs)

    def _inner_training_loop(
            self, batch_size=None, args=None, resume_from_checkpoint=None,
            trial=None, ignore_keys_for_eval=None
    ):
        # add task losses logging
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        # total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        total_train_batch_size = train_dataloader.batch_size_sum * \
                                 args.world_size * \
                                 args.gradient_accumulation_steps
        task_total_batch_sizes = [
            bsz * args.gradient_accumulation_steps * args.world_size for bsz in
            self.task_batch_sizes]
        if self.data_args.in_batch_negatives:
            task_effect_bszs = [b * args.world_size if
                                self.args.negatives_x_device else b for b in
                                self.per_device_task_batch_sizes]
            task_num_candidates = [b * n for b, n in zip(task_effect_bszs,
                                                         self.n_passages)]
        else:
            task_num_candidates = self.n_passages

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(
                    args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(
                    train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
                self.sharded_ddp is not None
                and self.sharded_ddp != ShardedDDPOption.SIMPLE
                or is_sagemaker_mp_enabled()
                or self.fsdp is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps,
                resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f" Tasks = {self.task_names}")
        logger.info(f" Task batch sizes per device = "
                    f"{self.per_device_task_batch_sizes}")
        logger.info(f" Task total batch sizes per device (w. parallel, "
                    f"distributed "
                    f"& accumulation) = {task_total_batch_sizes} ")
        logger.info(
            f" Task number candidates per query = {task_num_candidates}")
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (
                    num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(
                f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(
                        total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description(
                        "Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(
            trial) if self.hp_name is not None else None
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        logger.info(f'max steps {max_steps}')
        logger.info(f'state max steps {self.state.max_steps}')
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        tr_task_losses = torch.zeros(self.num_tasks).to(args.device)
        if not self.args.log_gnorm or self.args.weight_method == 'naive':
            tr_grad_norms = None
        else:
            tr_grad_norms = torch.zeros(
                self.num_tasks).to(self.args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._total_task_loss_scalars = [0.0 for _ in range(self.num_tasks)]
        # self._total_task_loss_scalars = torch.zeros(self.num_tasks).to(
        #     self.args.device)
        if not self.args.log_gnorm or self.args.weight_method == 'naive':
            self._total_grad_norm_scalars = None
        else:
            self._total_grad_norm_scalars = torch.zeros(self.num_tasks).to(
                self.args.device)
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state,
                                                            self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader,
                                            "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if version.parse(torch.__version__) < version.parse(
                        "1.11") or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)
        if self.args.hard_negative_mining:
            assert (num_train_epochs % self.args.epochs_per_hn) == 0, \
                "total num epochs should be divided by epochs per hard neg"
            target_epochs = (epochs_trained // self.args.epochs_per_hn
                             + 1) * self.args.epochs_per_hn
        else:
            target_epochs = num_train_epochs

        for epoch in range(epochs_trained, target_epochs):
            logger.info(f"target epoch stop {target_epochs}")
            if isinstance(train_dataloader, DataLoader) and isinstance(
                    train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(
                    train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [
                    args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args,
                                                                self.state,
                                                                self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args,
                                                                       self.state,
                                                                       self.control)

                if (
                        ((step + 1) % args.gradient_accumulation_steps != 0)
                        and args.local_rank != -1
                        and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step, tr_task_losses_step, tr_grads_norm_step \
                            = self.training_step(
                            model, inputs)
                else:
                    tr_loss_step, tr_task_losses_step, tr_grads_norm_step = \
                        self.training_step(
                            model, inputs)

                if (
                        args.logging_nan_inf_filter
                        and not is_torch_tpu_available()
                        and (
                        torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (
                            1 + self.state.global_step - self._globalstep_last_logged)
                    tr_task_losses /= tr_task_losses / (
                            1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step
                    tr_task_losses += tr_task_losses_step
                    # for i in range(self.num_tasks):
                    #     tr_task_losses[i] += tr_task_losses_step[i]
                if self.args.log_gnorm and tr_grads_norm_step is not None:
                    tr_grad_norms += tr_grads_norm_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients,
                                              scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(
                                    self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            xm.optimizer_step(self.optimizer)
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args,
                                                                     self.state,
                                                                     self.control)
                    self._maybe_log_save_metrics(tr_grad_norms, tr_task_losses,
                                                 tr_loss, model, trial,
                                                 epoch,
                                                 ignore_keys_for_eval)
                    # self._maybe_log_save_evaluate(tr_loss, model, trial, epoch,
                    #                               ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args,
                                                                        self.state,
                                                                        self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state,
                                                              self.control)
            # self._maybe_log_save_evaluate(tr_loss, model, trial, epoch,
            #                               ignore_keys_for_eval)
            self._maybe_log_save_metrics(tr_grad_norms, tr_task_losses, tr_loss,
                                         model, trial, epoch,
                                         ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info(
            "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step
        # self._total_task_loss_scalars += tr_task_losses
        for i in range(self.num_tasks):
            self._total_task_loss_scalars[i] += tr_task_losses[i].item()
        if not self.args.log_gnorm or self.args.weight_method == 'naive':
            task_grad_norms = None
        else:
            self._total_grad_norm_scalars += tr_grad_norms
            task_grad_norms = {k: l / self.state.global_step for k, l in zip(
                self.task_names, self._total_grad_norm_scalars)}
        task_train_losses = {k: l / self.state.global_step for k, l in zip(
            self.task_names, self._total_task_loss_scalars)}

        metrics = speed_metrics("train", start_time,
                                num_samples=num_train_samples,
                                num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss
        metrics["train_losses"] = task_train_losses
        if not self.args.log_gnorm or self.args.weight_method == 'naive':
            metrics['train_grad_norms'] = task_grad_norms

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)
        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state,
                                                          self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _maybe_log_save_metrics(self,
                                tr_grad_norms,
                                tr_task_losses,
                                tr_loss, model, trial, epoch,
                                ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()
            logs = {}
            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            # reset tr_loss to zero
            tr_loss -= tr_loss
            logs["loss"] = round(tr_loss_scalar / (
                    self.state.global_step - self._globalstep_last_logged),
                                 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalars = self._nested_gather(tr_task_losses[None]).mean(
                0).cpu().tolist()
            tr_task_losses -= tr_task_losses
            log_losses = [round(
                t_scalar / (self.state.global_step -
                            self._globalstep_last_logged), 4) for t_scalar in
                tr_loss_scalars]
            logs["losses"] = {k: l for k, l in zip(self.task_names, log_losses)}
            for i in range(self.num_tasks):
                self._total_task_loss_scalars[i] += tr_loss_scalars[i]
            if tr_grad_norms is not None:
                tr_norm_scalar = tr_grad_norms.item() if tr_grad_norms.dim() == 0 \
                    else tr_grad_norms.cpu().tolist()
                tr_grad_norms -= tr_grad_norms
                if tr_grad_norms.dim() == 0:
                    log_grad_norms = round(tr_norm_scalar / (
                            self.state.global_step - self._globalstep_last_logged),
                                           4)
                else:
                    log_norms = [round(
                        t_scalar / (self.state.global_step -
                                    self._globalstep_last_logged), 4) for
                        t_scalar
                        in tr_norm_scalar]
                    log_grad_norms = {k: l for k, l in
                                      zip(self.task_names, log_norms)}
                self._total_grad_norm_scalars += tr_norm_scalar
                logs['grad_norm'] = log_grad_norms
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state,
                                                         self.control)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        logger.info(f"Loading model from {resume_from_checkpoint}.")


class MultiTaskTBCallback(TensorBoardCallback):

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.tb_writer is None:
            self._init_summary_writer(args)

        if self.tb_writer is not None:
            logs = rewrite_logs(logs)
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, state.global_step)
                elif isinstance(v, dict):
                    self.tb_writer.add_scalars(k, v, state.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            self.tb_writer.flush()


def dist_gather_tensor(t, world_size, process_index):
    if t is None:
        return None
    t = t.contiguous()

    all_tensors = [torch.empty_like(t) for _ in range(world_size)]
    dist.all_gather(all_tensors, t)

    all_tensors[process_index] = t
    all_tensors = torch.cat(all_tensors, dim=0)

    return all_tensors


def gini(gw):
    # gw : T x p
    """Calculate the Gini indexes of gw along dim 0"""
    # Values cannot be 0:
    gw += 1e-12
    gw_sorted = torch.sort(gw, dim=0)[0]
    n = gw.size(0)
    # T  - > T x 1
    c = (2 * torch.arange(1, n + 1) - n - 1).unsqueeze(1)
    # 1 x p
    indexes = (c * gw_sorted).sum(0) / (n * gw_sorted.sum(0))
    return indexes


def major_entropy(gw):
    gw_max = torch.max(gw, dim=0)[0]
    ent = -gw_max.log()
    return ent
