import logging
import os
import sys

from taco.arguments import DataArguments
from taco.arguments import DenseEncodingArguments as EncodingArguments
from taco.arguments import ModelArguments
from taco.dataset import InferenceDataset
from taco.modeling import DenseModelForInference
from taco.retriever import Retriever
from taco.utils import save_as_trec
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser,\
    AutoProcessor

logger = logging.getLogger(__name__)


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

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if encoding_args.local_rank in [-1,
                                                           0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        encoding_args.local_rank,
        encoding_args.device,
        encoding_args.n_gpu,
        bool(encoding_args.local_rank != -1),
        encoding_args.fp16,
    )
    logger.info("Encoding parameters %s", encoding_args)
    logger.info("MODEL parameters %s", model_args)

    num_labels = 1
    _qry_model_path = os.path.join(
        model_args.model_name_or_path,
        f'query_model_{data_args.task_name}') if model_args.split_query_encoder \
        else model_args.model_name_or_path
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else _qry_model_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    model = DenseModelForInference.build(
        model_args=model_args,
        data_args=data_args,
        train_args=encoding_args,
        config=config,
        cache_dir=model_args.cache_dir,
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

    query_dataset = InferenceDataset.load(
        tokenizer=tokenizer,
        data_args=data_args,
        is_query=True,
        stream=False,
        is_image=data_args.is_image,
        processor=processor,
        cache_dir=model_args.cache_dir,
    )
    query_dataset.task_name = data_args.task_name
    retriever = Retriever.from_embeddings(model, encoding_args)
    result = retriever.retrieve(query_dataset)
    if encoding_args.local_process_index == 0:
        trec_save_dir = os.path.dirname(encoding_args.trec_save_path)
        if not os.path.exists(trec_save_dir):
            os.mkdir(trec_save_dir)
        save_as_trec(result, encoding_args.trec_save_path)


if __name__ == '__main__':
    main()
