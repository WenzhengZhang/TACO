import os
import sys

from taco.arguments import DataArguments
from taco.arguments import DenseEncodingArguments as EncodingArguments
from taco.arguments import ModelArguments
from taco.dataset import InferenceDataset
from taco.modeling import DenseModelForInference
from taco.retriever import Retriever
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, \
    AutoProcessor


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, EncodingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, encoding_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, encoding_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        encoding_args: EncodingArguments

    num_labels = 1
    _psg_model_path = os.path.join(
        model_args.model_name_or_path,
        'passage_model') if model_args.split_query_encoder else \
        model_args.model_name_or_path
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else _psg_model_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        # use_fast=False,
    )
    processor = None
    if data_args.is_image:
        try:
            processor = AutoProcessor.from_pretrained(
                model_args.processor_name if model_args.processor_name else
                model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
            )
        except ValueError:
            processor = None
    model = DenseModelForInference.build(
        model_args=model_args,
        data_args=data_args,
        train_args=encoding_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    corpus_dataset = InferenceDataset.load(
        tokenizer=tokenizer,
        data_args=data_args,
        is_query=False,
        stream=False,
        is_image=data_args.is_image,
        processor=processor,
        cache_dir=model_args.cache_dir
    )

    Retriever.build_embeddings(model, corpus_dataset, encoding_args)


if __name__ == '__main__':
    main()
