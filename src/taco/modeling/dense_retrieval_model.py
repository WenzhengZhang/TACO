import json
import os
import sys
import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
import torch.distributed as dist
import numpy as np
from ..loss import MultiLabelLoss
from ..utils import mean_pooling
import torch.nn.functional as F
from typing import Union

from transformers import AutoModel, BatchEncoding, PreTrainedModel, \
    T5EncoderModel
from transformers.modeling_outputs import ModelOutput

try:
    from transformers.adapters import AutoAdapterModel, AdapterSetup
except ImportError:
    print(
        'if use adapter please install transformer-adapter otherwise ignore this')
from typing import Optional, Dict, Union

from ..arguments import ModelArguments, DataArguments, \
    DenseTrainingArguments as TrainingArguments, DenseEncodingArguments as \
    EncodingArguments
from ..arguments import DenseEncodingArguments as EncodingArguments
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

"""
support models:

done: 1. standard DR model with in-batch negatives
done: 2. standard DR model with NCE style negatives
done: 3. support split query encoder option
done: 4. support query adapter option

"""


@dataclass
class DenseOutput(ModelOutput):
    q_reps: Tensor = None
    p_reps: Tensor = None
    loss: Tensor = None
    scores: Tensor = None


class LinearHead(nn.Module):
    def __init__(
            self,
            input_dim: int = 768,
            output_dim: int = 768,
    ):
        super(LinearHead, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.config = {'input_dim': input_dim, 'output_dim': output_dim}

    def forward(self, rep: Tensor = None):
        return self.linear(rep)

    @classmethod
    def load(cls, ckpt_dir: str):
        logger.info(f'Loading linear head from {ckpt_dir}')
        model_path = os.path.join(ckpt_dir, 'linear.pt')
        config_path = os.path.join(ckpt_dir, 'head_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        model = cls(**config)
        model.load_state_dict(torch.load(model_path))
        return model

    def save(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, 'linear.pt'))
        with open(os.path.join(save_path, 'head_config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)


# TODO: load and save query adapters
class DenseModel(nn.Module):
    def __init__(
            self,
            lm_q: Union[PreTrainedModel, nn.ModuleDict],
            lm_p: PreTrainedModel,
            head_q: nn.Module = None,
            head_p: nn.Module = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
    ):
        super().__init__()

        self.lm_q = lm_q
        self.lm_p = lm_p
        self.head_q = head_q
        self.head_p = head_p
        self.loss_fct = MultiLabelLoss(train_args.type_loss) if \
            train_args.multi_label else nn.CrossEntropyLoss(reduction='mean')

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args
        self.config = self.lm_p.config  # make deepspeed happy

        if train_args.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError(
                    'Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def check_nan_inf(self, t: torch.Tensor, location=""):
        if t.isnan().any():
            if self.train_args.skip_steps_with_Nan:
                logger.error(f"Error, NaN found in tensor @ {location}")
                return False
            else:
                raise RuntimeError(f"Error, NaN found in tensor @ {location}")
        if t.isinf().any():
            if self.train_args.skip_steps_with_Nan:
                logger.error(f"Error, NaN found in tensor @ {location}")
                return False
            else:
                raise RuntimeError(f"Error, Inf found in tensor @ {location}")
        return True

    def forward(
            self,
            query: Dict[str, Tensor] = None,
            passage: Dict[str, Tensor] = None,
            target: Tensor = None,
            n_passages: int = None,
            task_name: str = None
    ):

        q_hidden, q_reps = self.encode_query(query, task_name)
        p_hidden, p_reps = self.encode_passage(passage)

        if q_reps is None or p_reps is None:
            return DenseOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )

        # if self.training:
        if self.train_args.negatives_x_device:
            q_reps = self.dist_gather_tensor(q_reps)
            p_reps = self.dist_gather_tensor(p_reps)

        ### scale down scores before matmul bc we got a NaN after the matmul for T5Large+...
        if self.data_args.in_batch_negatives:
            scores = torch.matmul(q_reps, p_reps.transpose(0, 1))
        else:
            bsz = q_reps.size(0)
            # ( B x 1 x d ) ( B x d x C) --> B x 1 x C --> B x C
            scores = torch.matmul(q_reps.unsqueeze(1),
                                  p_reps.view(bsz, -1, p_reps.size(
                                      -1)).transpose(1, 2)).view(bsz, -1)

        if not self.check_nan_inf(scores, "DenseModel.forward(), scores"):
            return None
        # print(scores.shape)
        # scores = scores.view(effective_bsz, -1)  # ???
        if self.train_args.multi_label:
            assert target is not None
            if self.data_args.in_batch_negatives:
                target = torch.block_diag(*target)
        else:
            assert target is None
            if self.data_args.in_batch_negatives:
                target = torch.arange(
                    scores.size(0),
                    device=scores.device,
                    dtype=torch.long
                )
                n_passages = self.data_args.train_n_passages if n_passages is None \
                    else n_passages
                target = target * n_passages
            else:  # when use NCE we always put positive candidate at the index 0
                target = torch.zeros(
                    scores.size(0),
                    device=scores.device,
                    dtype=torch.long
                )

        loss = self.loss_fct(scores, target)
        if not self.check_nan_inf(loss, "DenseModel.forward(), loss"):
            return None

        if self.training and self.train_args.negatives_x_device:
            loss = loss * self.world_size  # counter average weight reduction
        return DenseOutput(
            loss=loss,
            scores=scores,
            # q_reps=q_reps,
            # p_reps=p_reps
        )

    def dense_encode(self, items, model, head, is_q=False):
        if items is None:
            return None, None
        items = BatchEncoding(items)
        if "T5" in type(model).__name__ and not self.model_args.encoder_only:
            decoder_input_ids = torch.zeros((items.input_ids.shape[0], 1),
                                            dtype=torch.long).to(
                items.input_ids.device)
            items_out = model(**items, decoder_input_ids=decoder_input_ids,
                              return_dict=True)
            hidden = items_out.last_hidden_state
            reps = hidden[:, 0, :]
        elif "CLIP" in type(model).__name__:
            reps = hidden = model.get_text_features(**items,
                                                    return_dict=True) if is_q else model.get_image_features(
                **items, return_dict=True)
        else:
            items_out = model(**items, return_dict=True)
            hidden = getattr(items_out, 'last_hidden_state')
            if self.model_args.pooling == "first":
                reps = hidden[:, 0, :]
            elif self.model_args.pooling == "mean":
                reps = mean_pooling(hidden, items.attention_mask)
            elif self.model_args.pooling == "no":
                reps = hidden
            else:
                raise ValueError(
                    "Unknown pooling type: {}".format(self.pooling))
        if head is not None:
            reps = head(reps)  # D * d
        if self.model_args.normalize:
            reps = F.normalize(reps, dim=1)
        return hidden, reps

    def encode_passage(self, psg):
        return self.dense_encode(psg, self.lm_p, self.head_p)

    def encode_query(self, qry, task_name: str = None):
        head_q = self.head_q
        if self.model_args.split_query_encoder:
            lm_q = self.lm_q[task_name]
            if self.model_args.add_linear_head:
                head_q = self.head_q[task_name]
        else:
            lm_q = self.lm_q
        if self.model_args.use_query_adapter:
            lm_q.set_active_adapters(None)
            lm_q.train_adapter([task_name])
            lm_q.set_active_adapters([task_name])
            lm_q.freeze_model(False)
        return self.dense_encode(qry, lm_q, head_q, True)

    @classmethod
    def build(
            cls,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: Union[TrainingArguments, EncodingArguments] = None,
            resume_path: str = None,
            **hf_kwargs,
    ):
        # load local
        head_q = head_p = None
        task_names = None
        if model_args.use_query_adapter:
            model_class = AutoAdapterModel
        else:
            model_class = T5EncoderModel if model_args.encoder_only else \
                AutoModel
        if model_args.split_query_encoder or model_args.use_query_adapter:
            task_names = train_args.task_names.split(',')
        if resume_path is not None:
            model_name_or_path = resume_path
        else:
            model_name_or_path = model_args.model_name_or_path
        if os.path.isdir(model_name_or_path):
            if model_args.untie_encoder:
                if model_args.split_query_encoder:
                    lm_q = {}
                    if model_args.add_linear_head:
                        head_q = {}
                    for task_name in task_names:
                        _qry_model_path = os.path.join(
                            model_name_or_path,
                            f'query_model_{task_name}')
                        if not os.path.exists(_qry_model_path):
                            _qry_model_path = model_name_or_path
                        logger.info(
                            f'loading query model weight from {_qry_model_path}')
                        lm_q_task = model_class.from_pretrained(
                            _qry_model_path,
                            **hf_kwargs
                        )
                        lm_q[task_name] = lm_q_task
                        _qry_head_path = os.path.join(
                            model_name_or_path,
                            f'query_head_{task_name}')
                        if model_args.add_linear_head:
                            head_q_task = LinearHead.load(_qry_head_path)
                            head_q[task_name] = head_q_task
                else:
                    _qry_model_path = os.path.join(
                        model_name_or_path,
                        'query_model')
                    if not os.path.exists(_qry_model_path):
                        _qry_model_path = model_name_or_path
                    logger.info(
                        f'loading query model weight from {_qry_model_path}')

                    lm_q = model_class.from_pretrained(
                        _qry_model_path,
                        **hf_kwargs
                    )
                    if model_args.use_query_adapter:
                        for task_name in task_names:
                            adapter_path = os.path.join(
                                model_name_or_path, f'query_adapter_'
                                                    f'{task_name}')
                            lm_q.load_adapter(adapter_path)
                    _qry_head_path = os.path.join(model_name_or_path,
                                                  'query_head')
                    if model_args.add_linear_head:
                        head_q = LinearHead.load(_qry_head_path)
                _psg_model_path = os.path.join(model_name_or_path,
                                               'passage_model')
                if not os.path.exists(_psg_model_path):
                    _psg_model_path = model_name_or_path

                _psg_head_path = os.path.join(model_name_or_path,
                                              'passage_head')

                logger.info(
                    f'loading passage model weight from {_psg_model_path}')
                lm_p = model_class.from_pretrained(
                    _psg_model_path,
                    **hf_kwargs
                )
                if model_args.add_linear_head:
                    head_p = LinearHead.load(_psg_head_path)
            else:
                lm_q = model_class.from_pretrained(
                    model_name_or_path,
                    **hf_kwargs)
                lm_p = lm_q
                if model_args.add_linear_head:
                    head_q = LinearHead.load(model_name_or_path)
                    head_p = head_q
        # load pre-trained
        else:
            lm_p = model_class.from_pretrained(model_args.model_name_or_path,
                                               **hf_kwargs)
            if model_args.add_linear_head:
                head_p = LinearHead(model_args.projection_in_dim,
                                    model_args.projection_out_dim)
            if model_args.split_query_encoder:
                lm_q = {task_name: copy.deepcopy(lm_p) for task_name in
                        task_names}
                if model_args.add_linear_head:
                    head_q = {task_name: copy.deepcopy(head_p) for task_name in
                              task_names}
            else:
                lm_q = copy.deepcopy(lm_p) if model_args.untie_encoder else lm_p
                if model_args.use_query_adapter and model_args.untie_encoder:
                    for task_name in task_names:
                        lm_q.add_adapter(task_name)
                if model_args.add_linear_head:
                    head_q = copy.deepcopy(head_p) if model_args.untie_encoder \
                        else head_p
        if model_args.split_query_encoder:
            lm_q = nn.ModuleDict(lm_q)
            if model_args.add_linear_head:
                head_q = nn.ModuleDict(head_q)

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            head_q=head_q,
            head_p=head_p,
            model_args=model_args,
            data_args=data_args,
            train_args=train_args,
        )
        return model

    def save(self, output_dir: str):
        task_names = None
        if self.model_args.split_query_encoder:
            task_names = self.train_args.task_names.split(',')
        if self.model_args.untie_encoder:
            if self.model_args.split_query_encoder:
                for task_name in task_names:
                    os.makedirs(os.path.join(output_dir, f'query_model_'
                                                         f'{task_name}'),
                                exist_ok=True)
                    self.lm_q[task_name].save_pretrained(
                        os.path.join(output_dir, f'query_model_{task_name}'))
                    if self.model_args.add_linear_head:
                        self.head_q[task_name].save(
                            os.path.join(output_dir, f'query_head_{task_name}')
                        )
            else:
                os.makedirs(
                    os.path.join(output_dir, 'query_model'), exist_ok=True)
                self.lm_q.save_pretrained(
                    os.path.join(output_dir, 'query_model'))
                if self.model_args.use_query_adapter:
                    for task_name in task_names:
                        adapter_path = os.path.join(output_dir,
                                                    f'query_adapter_{task_name}')
                        self.lm_q.save_adapter(adapter_path, task_name)
                if self.model_args.add_linear_head:
                    self.head_q.save(os.path.join(output_dir, 'query_head'))
            os.makedirs(os.path.join(output_dir, 'passage_model'),
                        exist_ok=True)
            self.lm_p.save_pretrained(os.path.join(output_dir, 'passage_model'))
            if self.model_args.add_linear_head:
                self.head_p.save(os.path.join(output_dir, 'passage_head'))
        else:
            self.lm_q.save_pretrained(output_dir)
            if self.model_args.add_linear_head:
                self.head_q.save(output_dir)

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors


class DenseModelForInference(DenseModel):

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def encode_passage(self, psg):
        return super(DenseModelForInference, self).encode_passage(psg)

    @torch.no_grad()
    def encode_query(self, qry, task_name=None):
        head_q = self.head_q
        if self.model_args.split_query_encoder:
            lm_q = self.lm_q[task_name]
            if self.model_args.add_linear_head:
                head_q = self.head_q[task_name]
        else:
            lm_q = self.lm_q
        if self.model_args.use_query_adapter:
            lm_q.set_active_adapters(None)
            lm_q.set_active_adapters([task_name])
            lm_q.freeze_model(True)
        return self.dense_encode(qry, lm_q, head_q, True)

    def forward(
            self,
            query: Dict[str, Tensor] = None,
            passage: Dict[str, Tensor] = None,
            target: Tensor = None,
            n_passages: int = None,
            task_name: str = None
    ):
        q_hidden, q_reps = self.encode_query(query, task_name)
        p_hidden, p_reps = self.encode_passage(passage)
        return DenseOutput(q_reps=q_reps, p_reps=p_reps)
