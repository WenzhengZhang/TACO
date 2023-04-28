import logging
import os
import sys
from taco.arguments import DataArguments
from taco.arguments import DenseTrainingArguments as TrainingArguments
from taco.arguments import ModelArguments
from taco.dataset import MappingDRTrainDataset
from taco.modeling import DenseModel
from taco.trainer import MTDenseTrainer, MultiTaskTBCallback
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed
from taco.utils import get_task_hps

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Lr decay while iterative training with hard negative
    if not training_args.hard_negative_mining:
        for iter in range(model_args.iter_num):
            training_args.learning_rate *= model_args.decay_rate

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1,
                                                           0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    num_labels = 1

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else
        model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    model = DenseModel.build(
        model_args,
        data_args,
        training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    # multi-task dataset
    train_files = data_args.mt_train_paths.split(',')
    val_files = data_args.mt_eval_paths.split(',')
    task_names = training_args.task_names.split(',')
    num_tasks = len(task_names)
    q_max_lens = get_task_hps(data_args.q_max_lens, num_tasks)
    p_max_lens = get_task_hps(data_args.p_max_lens, num_tasks)
    n_passages = get_task_hps(data_args.mt_train_n_passages, num_tasks)
    train_dataset = [MappingDRTrainDataset(
        tokenizer, data_args,
        q_max_len=q_max_lens[i], p_max_len=p_max_lens[i],
        n_passages=n_passages[i], multi_label=training_args.multi_label,
        shuffle_seed=training_args.seed,
        cache_dir=data_args.data_cache_dir or model_args.cache_dir,
        task_name=task_names[i],
        data_file=train_files[i]
    ) for i in range(num_tasks)]
    eval_dataset = [MappingDRTrainDataset(
        tokenizer, data_args,
        q_max_len=q_max_lens[i], p_max_len=p_max_lens[i],
        n_passages=n_passages[i],
        is_eval=True,
        multi_label=training_args.multi_label,
        shuffle_seed=training_args.seed,
        cache_dir=data_args.data_cache_dir or model_args.cache_dir,
        task_name=task_names[i],
        data_file=val_files[i]
    ) for i in range(num_tasks)]
    tb_callback = MultiTaskTBCallback()
    trainer = MTDenseTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[tb_callback],
        data_args=data_args
    )
    for train_task_set in train_dataset:
        train_task_set.trainer = trainer
    if training_args.resume_from_checkpoint is not None and \
            training_args.resume_from_checkpoint != False and \
            training_args.resume_from_checkpoint != "False":
        trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
