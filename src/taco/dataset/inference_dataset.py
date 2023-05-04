from datasets import load_dataset
from torch.utils.data import IterableDataset, Dataset
from transformers import PreTrainedTokenizer, AutoProcessor, ProcessorMixin
import os
from typing import Union, List, Callable
from PIL import Image
import datasets

from ..arguments import DataArguments
from ..utils import find_all_markers, fill_template, get_idx

"""
support features:

done: 1. stream and map dataset
done: 2. add task prefix
3. beir dataset
"""


class InferenceMapDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments,
                 is_query: bool = False, cache_dir: str = None):
        super(InferenceMapDataset, self).__init__()
        self.cache_dir = cache_dir
        self.processed_data_path = data_args.processed_data_path
        self.data_files = [data_args.query_path] if is_query else [
            data_args.corpus_path]
        self.tokenizer = tokenizer
        self.max_len = data_args.q_max_len if is_query else data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        # nq query template should be <question>
        self.template = data_args.query_template if is_query else data_args.doc_template
        self.all_markers = find_all_markers(self.template)
        self.is_query = is_query
        self.data_args = data_args
        if is_query:
            if self.data_args.add_query_task_prefix:
                self.task_prefix = self.get_task_prefix(data_args.task_name)
        else:
            if self.data_args.add_passage_task_prefix:
                self.task_prefix = self.get_task_prefix(data_args.task_name)

    def get_task_prefix(self, task_name):
        task_token = '<extra_id_1>'
        task_prefix = ' '.join([task_name, task_token])
        return task_prefix

    @classmethod
    def load(cls, tokenizer: PreTrainedTokenizer, data_args: DataArguments,
             is_query: bool = False, cache_dir: str = None):
        data_files = [data_args.query_path] if is_query else [
            data_args.corpus_path]
        ext = os.path.splitext(data_files[0])[1]
        if ext == ".jsonl":
            return JsonlMapDataset(tokenizer, data_args, is_query, cache_dir)
        elif ext in [".tsv", ".txt", ".csv"]:
            return TsvMapDataset(tokenizer, data_args, is_query, cache_dir)
        else:
            raise ValueError(
                "Unsupported dataset file extension {}".format(ext))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]
        example_id = get_idx(example)
        full_text = self.template
        # For mind dataset where query is a list of titles
        for marker in self.all_markers:
            full_text = full_text.replace("<{}>".format(marker),
                                          example[marker] if example[
                                                                 marker] is not None else "")
        if (self.is_query and self.data_args.add_query_task_prefix) or (
                self.data_args.add_passage_task_prefix and not self.is_query):
            full_text = ' '.join([self.task_prefix, full_text])
        tokenized = self.tokenizer(full_text, padding='max_length',
                                   truncation=True, max_length=self.max_len)

        return {"text_id": example_id, **tokenized}


class JsonlMapDataset(InferenceMapDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments,
                 is_query: bool = False, cache_dir: str = None):
        super(JsonlMapDataset, self).__init__(tokenizer, data_args, is_query,
                                              cache_dir)
        self.dataset = load_dataset("json", data_files=self.data_files,
                                    cache_dir=cache_dir)["train"]
        sample = self.dataset[0]
        self.all_columns = sample.keys()


class TsvMapDataset(InferenceMapDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments,
                 is_query: bool = False, cache_dir: str = None):
        super(TsvMapDataset, self).__init__(tokenizer, data_args, is_query,
                                            cache_dir)
        self.all_columns = data_args.query_column_names if is_query else data_args.doc_column_names
        if self.all_columns is not None:
            self.all_columns = self.all_columns.split(',')
        # set doc column name to be None for nq/trivia
        self.dataset = load_dataset(
            "csv",
            data_files=self.data_files,
            column_names=self.all_columns,
            delimiter='\t',
            cache_dir=cache_dir,
            keep_default_na=False
        )["train"]
        if self.all_columns is None:
            sample = self.dataset[0]
            self.all_columns = sample.keys()
        if 'id' not in self.all_columns:
            ids = list(range(len(self.dataset)))
            self.dataset = self.dataset.add_column('id', ids)
            self.all_columns = ['id'] + self.all_columns


class InferenceDataset:

    def __init__(
            self,
            data_args: DataArguments,
            tokenizer: PreTrainedTokenizer = None,
            processor: ProcessorMixin = None,
            is_query: bool = False,
            full_tokenization: bool = True,
            mode: str = "processed",
            filter_fn: Callable = lambda x: True,
            cache_dir: str = None,
            is_image: bool = False,
    ):
        self.cache_dir = cache_dir
        self.is_query = is_query
        self.data_files = [data_args.query_path] if is_query else [
            data_args.corpus_path]
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_len = data_args.q_max_len if self.is_query else data_args.p_max_len
        self.is_image = is_image

        self.template = data_args.query_template if self.is_query else data_args.doc_template

        self.full_tokenization = full_tokenization
        modes = ["raw", "dict_processed", "processed"]
        if mode not in modes:
            raise ValueError(f"mode must be one of {modes}")
        self.mode = mode
        self.filter_fn = filter_fn
        self._prepare_data(data_args)

        if not self.is_image:
            self.all_markers = find_all_markers(
                self.template) if data_args.all_markers is None else data_args.all_markers.split(
                ",")
        self.data_args = data_args
        self.task_name = data_args.task_name
        if is_query:
            if self.data_args.add_query_task_prefix:
                self.task_prefix = self.get_task_prefix(data_args.task_name)
        else:
            if self.data_args.add_passage_task_prefix:
                self.task_prefix = self.get_task_prefix(data_args.task_name)

    def _prepare_data(self, data_args):
        raise NotImplementedError

    def get_task_prefix(self, task_name):
        task_token = '<extra_id_1>'
        task_prefix = ' '.join([task_name, task_token])
        return task_prefix

    @classmethod
    def load(
            cls,
            data_args: DataArguments,
            tokenizer: PreTrainedTokenizer = None,
            processor: ProcessorMixin = None,
            is_query: bool = False,
            full_tokenization: bool = True,
            mode: str = "processed",
            stream: bool = False,
            filter_fn: Callable = lambda x: True,
            cache_dir: str = None,
            is_image: bool = False,
    ):
        data_files = [data_args.query_path] if is_query else [
            data_args.corpus_path]
        ext = os.path.splitext(data_files[0])[1]
        ext_to_cls = {
            ".jsonl": StreamJsonlDataset if stream else MappingJsonlDataset,
            ".tsv": StreamTsvDataset if stream else MappingTsvDataset,
            ".txt": StreamTsvDataset if stream else MappingTsvDataset,
        }
        cls_ = ext_to_cls.get(ext, None) if ext != "" else StreamImageDataset
        if cls_ is None:
            raise ValueError(
                "Unsupported dataset file extension {}".format(ext))
        return cls_(
            tokenizer=tokenizer,
            processor=processor,
            data_args=data_args,
            is_query=is_query,
            full_tokenization=full_tokenization,
            mode=mode,
            filter_fn=filter_fn,
            cache_dir=cache_dir,
            is_image=is_image
        )

    def _tokenize(self, example: str):
        return self.tokenizer(
            example,
            add_special_tokens=self.full_tokenization,
            padding='max_length' if self.full_tokenization else False,
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=self.full_tokenization,
            return_token_type_ids=False
        )

    def process_one(self, example):
        if self.is_image:
            path = example["image"]["path"]
            img = Image.open(path)
            processed = self.processor(images=img)
            name = os.path.basename(path).split(".")[0]
            return {"text_id": name,
                    "pixel_values": processed["pixel_values"][0]}
        elif self.mode == "raw":
            return example
        elif self.mode == "dict_processed":
            example_id = get_idx(example)
            tokenized = {}
            for marker in self.all_markers:
                tokenized[marker] = dict(
                    self._tokenize(example[marker])) if (
                        marker in example and example[
                    marker] is not None) else None
            if (self.is_query and self.data_args.add_query_task_prefix) or (
                    self.data_args.add_passage_task_prefix and not self.is_query):
                tokenized["task_prefix"] = self._tokenize(self.task_prefix)
            return {"text_id": example_id, **tokenized}
        else:
            example_id = get_idx(example)
            # full_text = self.template
            # for marker in self.all_markers:
            #     full_text = full_text.replace(
            #         "<{}>".format(marker),
            #         example[marker] if example[marker] is not None else "")

            full_text = fill_template(self.template, example, self.all_markers,
                                      allow_not_found=True)
            if (self.is_query and self.data_args.add_query_task_prefix) or (
                    self.data_args.add_passage_task_prefix and not self.is_query):
                full_text = ' '.join([self.task_prefix, full_text])
            tokenized = self._tokenize(full_text)
            return {"text_id": example_id, **tokenized}


class StreamInferenceDataset(IterableDataset):

    def __iter__(self):
        real_batch_size = self.batch_size * self.num_processes
        process_slice = range(self.process_index * self.batch_size,
                              (self.process_index + 1) * self.batch_size)

        current_batch = []
        for element in self.dataset:
            current_batch.append(element)
            # Wait to have a full batch before yielding elements.
            if len(current_batch) == real_batch_size:
                for i in process_slice:
                    yield self.process_one(current_batch[i])
                current_batch = []

        if len(current_batch) > 0:
            for i in process_slice:
                if i < len(current_batch):
                    yield self.process_one(current_batch[i])


class MappingInferenceDataset(Dataset):

    def __getitem__(self, index):
        return self.process_one(self.dataset[index])

    def get_raw(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class StreamJsonlDataset(StreamInferenceDataset, InferenceDataset):

    def _prepare_data(self, data_args):
        self.dataset = load_dataset(
            "json",
            data_files=self.data_files,
            streaming=True,
            cache_dir=self.cache_dir
        )["train"].filter(self.filter_fn)
        sample = list(self.dataset.take(1))[0]
        self.all_columns = sample.keys()


class MappingJsonlDataset(MappingInferenceDataset, InferenceDataset):

    def _prepare_data(self, data_args):
        self.dataset = load_dataset(
            "json",
            data_files=self.data_files,
            streaming=False,
            cache_dir=self.cache_dir
        )["train"].filter(self.filter_fn)
        sample = self.dataset[0]
        self.all_columns = sample.keys()
        # self.dataset = {}
        # for item in hf_dataset:
        #     self.dataset[get_idx(item)] = item


class StreamTsvDataset(StreamInferenceDataset, InferenceDataset):

    def _prepare_data(self, data_args):
        self.all_columns = data_args.query_column_names if self.is_query else data_args.doc_column_names
        if self.all_columns is not None:
            self.all_columns = self.all_columns.split(',')
        self.dataset = load_dataset(
            "csv",
            data_files=self.data_files,
            streaming=True,
            column_names=self.all_columns,
            delimiter='\t',
            cache_dir=self.cache_dir,
            keep_default_na=False
        )["train"].filter(self.filter_fn)


class MappingTsvDataset(MappingInferenceDataset, InferenceDataset):

    def _prepare_data(self, data_args):
        self.all_columns = data_args.query_column_names if self.is_query else data_args.doc_column_names
        if self.all_columns is not None:
            self.all_columns = self.all_columns.split(',')
        self.dataset = load_dataset(
            "csv",
            data_files=self.data_files,
            streaming=False,
            column_names=self.all_columns,
            delimiter='\t',
            cache_dir=self.cache_dir,
            keep_default_na=False
        )["train"].filter(self.filter_fn)
        if self.all_columns is None:
            sample = self.dataset[0]
            self.all_columns = sample.keys()
        # if 'id' not in self.all_columns:
        #     ids = list(range(len(self.dataset)))
        #     self.dataset = self.dataset.add_column('id', ids)
        #     self.all_columns = ['id'] + self.all_columns

        # self.dataset = {}
        # for item in hf_dataset:
        #     self.dataset[get_idx(item)] = item


class StreamImageDataset(StreamInferenceDataset, InferenceDataset):

    def _prepare_data(self, data_args):
        self.is_image = True
        self.dataset = load_dataset(
            self.data_files[0],
            split="train",
            streaming=True,
        )
        self.dataset = self.dataset.cast_column("image",
                                                datasets.Image(decode=False))
