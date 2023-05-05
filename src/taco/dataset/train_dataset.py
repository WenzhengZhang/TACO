# adapted from openmatch

import glob
import os
import random
from typing import List, Tuple, Callable, Dict, Union

from datasets import load_dataset
from torch.utils.data import IterableDataset, Dataset
from transformers import (BatchEncoding, DataCollatorWithPadding,
                          PreTrainedTokenizer)
from transformers.data.data_collator import DefaultDataCollator

from ..arguments import DataArguments
from ..trainer import DenseTrainer
from itertools import cycle
import numpy as np
import math
from ..utils import shuffle_cycle

"""
Support features:
done: 1. standard DR dataset (stream and map style)
done: 2. standard multi-task dataset
3. beir dataset and qa dataset
done 4. in-batch style and NCE style negatives, also modify corresponding 
collator
done 5. multi-label dataset

"""


# data collator to support multi_label
# TODO: build_hn to support random negatives sampling


class TrainDatasetBase:
    '''
    Abstract base class for all train datasets.\n
    All future dataset ABCs would subclass this and `(Iterable)Dataset`.
    '''

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            data_args: DataArguments,
            q_max_len: int = None,
            p_max_len: int = None,
            n_passages: int = None,
            trainer: DenseTrainer = None,
            is_eval: bool = False,
            multi_label: bool = False,
            shuffle_seed: int = None,
            cache_dir: str = None,
            task_name: str = None,
            data_file: str = None
    ) -> None:
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.q_max_len = data_args.q_max_len if q_max_len is None else q_max_len
        self.p_max_len = data_args.p_max_len if p_max_len is None else p_max_len
        self.n_passages = data_args.train_n_passages if n_passages is None \
            else n_passages
        self.trainer = trainer
        self.is_eval = is_eval
        self.multi_label = multi_label
        self.task_name = task_name
        self._prepare_data(data_args, shuffle_seed, cache_dir, data_file)
        if self.data_args.add_query_task_prefix or \
                self.data_args.add_passage_task_prefix:
            self.task_prefix = self.get_task_prefix(task_name)

    def get_task_prefix(self, task_name):
        task_token = '<extra_id_1>'
        task_name = ' '.join([task_name, task_token])
        task_prefix = self.tokenizer.encode(
            task_name,
            add_special_tokens=False)
        return task_prefix

    def _prepare_data(self, data_args, shuffle_seed, cache_dir, data_file):
        if data_file is not None:
            self.data_files = [data_file]
        elif not self.is_eval:
            self.data_files = [
                data_args.train_path] if data_args.train_dir is None else glob.glob(
                os.path.join(data_args.train_dir, "*.jsonl"))
        else:
            self.data_files = [data_args.eval_path]

    def get_process_fn(self, epoch, hashed_seed):
        raise NotImplementedError


class StreamTrainDatasetMixin(IterableDataset, TrainDatasetBase):

    def _prepare_data(self, data_args, shuffle_seed, cache_dir, data_file):
        super()._prepare_data(data_args, shuffle_seed, cache_dir, data_file)
        self.dataset = load_dataset(
            "json", data_files=self.data_files, streaming=True,
            cache_dir=cache_dir)["train"]
        self.dataset = self.dataset.shuffle(
            seed=shuffle_seed,
            buffer_size=10_000) if shuffle_seed is not None else self.dataset
        sample = list(self.dataset.take(1))[0]
        self.all_columns = list(sample.keys())
        self.all_columns.remove('query')

    def __len__(self):
        concat_filenames = " ".join(self.data_files)
        count = 0
        with os.popen("wc -l {}".format(concat_filenames)) as f:
            for line in f:
                lc, filename = line.strip().split()
                lc = int(lc)
                if filename != "total":
                    count += lc
        return count

    def __iter__(self):
        if not self.is_eval:
            epoch = int(self.trainer.state.epoch)
            _hashed_seed = hash(self.trainer.args.seed)
            self.dataset.set_epoch(epoch)
            return iter(
                self.dataset.map(self.get_process_fn(epoch, _hashed_seed),
                                 remove_columns=self.all_columns))
        return iter(self.dataset.map(self.get_process_fn(0, None),
                                     remove_columns=self.all_columns))


class MappingTrainDatasetMixin(Dataset, TrainDatasetBase):

    def _prepare_data(self, data_args, shuffle_seed, cache_dir, data_file):
        super()._prepare_data(data_args, shuffle_seed, cache_dir, data_file)
        self.dataset = load_dataset(
            "json", data_files=self.data_files, streaming=False,
            cache_dir=cache_dir)["train"]
        sample = self.dataset[0]
        self.all_columns = list(sample.keys())
        self.all_columns.remove('query')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        group = self.dataset[index]
        if not self.is_eval:
            epoch = int(self.trainer.state.epoch)
            _hashed_seed = hash(index + self.trainer.args.seed)
            return self.get_process_fn(epoch, _hashed_seed)(group)
        return self.get_process_fn(0, None)(group)


class DRTrainDataset(TrainDatasetBase):

    def create_one_example(self, text_encoding: List[int],
                           is_query=False) -> BatchEncoding:
        if (is_query and self.data_args.add_query_task_prefix) or \
                (not is_query and self.data_args.add_passage_task_prefix):
            text_encoding = self.task_prefix + text_encoding
        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.q_max_len if is_query else self.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def get_process_fn(self, epoch, hashed_seed):

        def process_fn(example):
            qry = example['query']
            encoded_query = self.create_one_example(qry, is_query=True)
            encoded_passages = []
            group_positives = example['positives']
            group_negatives = example['negatives']
            target = []
            if self.multi_label:
                pos_psg = group_positives
                negative_size = self.n_passages - len(pos_psg)
                for pos in pos_psg:
                    target.append(1)
                    encoded_passages.append(self.create_one_example(pos))
            elif self.data_args.positive_passage_no_shuffle or hashed_seed is \
                    None:
                pos_psg = group_positives[0]
                negative_size = self.n_passages - 1
                encoded_passages.append(self.create_one_example(pos_psg))
            else:
                pos_psg = group_positives[
                    (hashed_seed + epoch) % len(group_positives)]
                negative_size = self.n_passages - 1
                encoded_passages.append(self.create_one_example(pos_psg))
            if not self.data_args.add_rand_negs or 'random_negatives' not in \
                    example:
                rand_negatives = []
                num_hard = negative_size
            else:
                num_rand = int(negative_size * self.data_args.rands_ratio)
                num_hard = negative_size - num_rand
                rand_negatives = example['random_negatives'][:num_rand]
            if len(group_negatives) < num_hard:
                if hashed_seed is not None:
                    hard_negatives = random.choices(group_negatives,
                                                    k=num_hard)
                else:
                    hard_negatives = [x for x in group_negatives]
                    hard_negatives = hard_negatives * 2
                    hard_negatives = hard_negatives[:num_hard]
            elif self.n_passages == 1:
                hard_negatives = []
            elif self.data_args.negative_passage_no_shuffle:
                hard_negatives = group_negatives[:num_hard]
            else:
                _offset = epoch * num_hard % len(group_negatives)
                hard_negatives = [x for x in group_negatives]
                if hashed_seed is not None:
                    random.Random(hashed_seed).shuffle(hard_negatives)
                hard_negatives = hard_negatives * 2
                hard_negatives = hard_negatives[_offset: _offset + num_hard]
            negs = hard_negatives + rand_negatives
            assert len(negs) == negative_size
            for neg_psg in negs:
                encoded_passages.append(self.create_one_example(neg_psg))

            assert len(encoded_passages) == self.n_passages
            model_inputs = {"query": encoded_query,
                            "passages": encoded_passages}
            if self.multi_label:
                target += [0] * negative_size
                model_inputs["target"] = target

            return model_inputs

        return process_fn


class StreamDRTrainDataset(StreamTrainDatasetMixin, DRTrainDataset):
    pass


class MappingDRTrainDataset(MappingTrainDatasetMixin, DRTrainDataset):
    pass


class MultiTaskDataLoader:

    def __init__(self, single_loaders, up_sample=True):
        self.single_loaders = single_loaders
        self.num_batches = [len(loader) for loader in self.single_loaders]
        # need this for hf trainer to compute num examples
        bsz_sum = sum(loader.batch_size for loader in single_loaders)
        # min_idx = np.argmin(self.num_batches)
        # max_idx = np.argmax(self.num_batches)
        # idx = max_idx if up_sample else min_idx
        # drop_last = single_loaders[idx].drop_last
        num_examples = sum(len(loader.dataset) for loader in single_loaders)
        # num_examples = len(
        #     single_loaders[idx].dataset) * bsz_sum / self.num_batches[
        #                    idx]
        # num_examples = math.floor(num_examples) if drop_last else math.ceil(
        #     num_examples)
        self.dataset = [None] * num_examples
        self.up_sample = up_sample
        self.batch_size_sum = bsz_sum

    def __len__(self):
        length = max(self.num_batches) if self.up_sample else min(
            self.num_batches)
        return length

    def __iter__(self):
        # output batch of different tasks, might have different batch size
        if self.up_sample:
            return iter(zip(*(shuffle_cycle(loader) if len(loader) < max(
                self.num_batches) else loader for loader in
                              self.single_loaders)))
        return iter(zip(*(loader for loader in self.single_loaders)))
