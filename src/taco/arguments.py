import os
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    target_model_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained reranker target model"}
    )

    config_name: Optional[str] = field(
        default=None, metadata={
            "help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    # modeling
    untie_encoder: bool = field(
        default=False,
        metadata={"help": "no weight sharing between qry passage encoders"}
    )

    # out projection
    add_pooler: bool = field(default=False)
    projection_in_dim: int = field(default=768)
    projection_out_dim: int = field(default=768)

    # for Jax training
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one "
                    "of `[float32, float16, bfloat16]`. "
        },
    )

    use_t5: bool = field(
        default=False,
        metadata={"help": "Whether to use T5 model"}
    )

    use_t5_decoder: bool = field(
        default=False,
        metadata={"help": "Whether to use T5 decoder"}
    )

    # for trainning with hard negative
    iter_num: Optional[int] = field(
        default=0, metadata={
            "help": "Iteration of hard negative generation, used to decay learning rate"}
    )

    decay_rate: Optional[float] = field(
        default=0.6, metadata={"help": "Decay learning rate"}
    )
    head_model_dim: int = field(default=768, metadata={
        "help": "head model dim"
    })
    prune_model: bool = field(default=False, metadata={
        "help": "prune model"
    })
    prune_indices_path: str = field(default=None, metadata={
        "help": "prune indices path"
    })
    entropy_path: str = field(default=None, metadata={
        "help": "entropy saving and loading path"
    })
    task_assign_path: str = field(default=None, metadata={
        "help": "task assign saving and loading path"
    })
    encoder_only: bool = field(
        default=False,
        metadata={"help": "use T5 encoder only"}
    )
    pooling: str = field(
        default='first',
        metadata={"help": "pooling strategy for getting query and passage "
                          "embedding (first, mean, no)"}
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "normalize query/passage embedding?"}
    )
    split_query_encoder: bool = field(
        default=False,
        metadata={"help": "split query encoder for multi_task retrieval"}
    )
    add_linear_head: bool = field(
        default=False,
        metadata={"help": "add linear head?"}
    )
    use_query_adapter: bool = field(
        default=False,
        metadata={"help": "use query adapter for multi_task retrieval"}
    )
    processor_name: str = field(
        default=None,
        metadata={"help": "image processor name"}
    )


@dataclass
class DataArguments:
    add_query_task_prefix: bool = field(default=False, metadata={
        "help": "add task prefix for query"})
    add_passage_task_prefix: bool = field(default=False, metadata={
        "help": "add task prefix for passage"
    })
    mt_train_paths: str = field(default=None, metadata={
        "help": "train data paths for multi-task training"})
    mt_eval_paths: str = field(default=None, metadata={
        "help": "eval data paths for multi-task training"})
    q_max_lens: str = field(default='32,32,512', metadata={
        "help": "query lengths for tasks"})
    p_max_lens: str = field(default='128,256,32', metadata={
        "help": "passage lengths for tasks"})
    over_sample: bool = field(
        default=True,
        metadata={"help": "over sample small dataset"})
    train_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    train_path: str = field(
        default=None, metadata={"help": "Path to single train file"}
    )
    eval_path: str = field(
        default=None, metadata={"help": "Path to eval file"}
    )
    query_path: str = field(
        default=None, metadata={"help": "Path to query file"}
    )
    task_name: str = field(
        default=None, metadata={"help": "task name for query"}
    )
    corpus_path: str = field(
        default=None, metadata={"help": "Path to corpus file"}
    )
    data_dir: str = field(
        default=None, metadata={"help": "Path to data directory"}
    )
    data_path: str = field(
        default=None, metadata={"help": "Path to the single data file"}
    )
    processed_data_path: str = field(
        default=None, metadata={"help": "Path to processed data directory"}
    )
    dataset_name: str = field(
        default=None, metadata={"help": "huggingface dataset name"}
    )
    data_name: str = field(
        default=None, metadata={'help': 'nq,marco,mind'}
    )
    passage_field_separator: str = field(default=' ')
    dataset_proc_num: int = field(
        default=12,
        metadata={"help": "number of proc used in dataset preprocess"}
    )
    train_n_passages: int = field(
        default=8,
        metadata={"help": "train n passages for dense retrieval"}
    )
    mt_train_n_passages: str = field(
        default='8',
        metadata={"help": "train n passages for multi_task retrieval"}
    )
    positive_passage_no_shuffle: bool = field(
        default=False,
        metadata={"help": "always use the first positive passage"})
    negative_passage_no_shuffle: bool = field(
        default=False,
        metadata={"help": "always use the first negative passages"})

    encode_in_path: List[str] = field(default=None, metadata={
        "help": "Path to data to encode"})

    encode_is_qry: bool = field(default=False)
    encode_num_shard: int = field(default=1)
    encode_shard_index: int = field(default=0)

    q_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    p_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    data_cache_dir: Optional[str] = field(
        default=None, metadata={
            "help": "Where do you want to store the data downloaded from huggingface"}
    )

    query_template: str = field(
        default="<text>",
        metadata={"help": "template for query"}
    )
    query_column_names: str = field(
        default=None,
        metadata={"help": "column names for the tsv data format"}
    )
    doc_template: str = field(
        default="Title: <title> Text: <text>",
        metadata={"help": "template for doc"}
    )
    doc_column_names: str = field(
        default=None,
        metadata={"help": "column names for the tsv data format"}
    )
    in_batch_negatives: bool = field(
        default=True,
        metadata={"help": "use in batch negative or hard-nce style negative"}
    )
    rands_ratio: float = field(
        default=0.5,
        metadata={"help": "random negatives fraction in nce"}
    )
    all_markers: str = field(
        default=None,
        metadata={"help": "all markers in the template"}
    )
    up_sample: bool = field(
        default=True,
        metadata={"help": "up-sample smaller dataset for multi_task training"}
    )
    is_image: bool = field(
        default=False,
        metadata={"help":"is image retrieval?"}
    )
    add_rand_negs: bool = field(
        default=False,
        metadata={"help":"add random negatives like hard-nce?"}
    )


    def __post_init__(self):
        pass
        # if self.dataset_name is not None:
        #     info = self.dataset_name.split('/')
        #     self.dataset_split = info[-1] if len(info) == 3 else 'train'
        #     self.dataset_name = "/".join(info[:-1]) if len(info) == 3 else '/'.join(info)
        #     self.dataset_language = 'default'
        #     if ':' in self.dataset_name:
        #         self.dataset_name, self.dataset_language = self.dataset_name.split(':')
        # else:
        #     self.dataset_name = 'json'
        #     self.dataset_split = 'train'
        #     self.dataset_language = 'default'
        # if self.train_dir is not None:
        #     files = os.listdir(self.train_dir)
        #     self.train_path = [
        #         os.path.join(self.train_dir, f)
        #         for f in files
        #         if f.endswith('jsonl') or f.endswith('json')
        #     ]
        # else:
        #     self.train_path = None


@dataclass
class DenseTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    negatives_x_device: bool = field(default=False, metadata={
        "help": "share negatives across devices"})
    do_encode: bool = field(default=False,
                            metadata={"help": "run the encoding loop"})

    grad_cache: bool = field(default=False,
                             metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=4)
    gc_p_chunk_size: int = field(default=32)
    multi_mix_temp: float = field(default=1.0)
    beta_taco: float = field(
        default=0.5, metadata={"help": "emphasis pow"}
    )
    beta_gn: float = field(
        default=0.5, metadata={"help": "emphasis pow"}
    )
    beta_cgd: float = field(
        default=0.5, metadata={"help": "emphasis pow"}
    )
    tau_taco: float = field(default=20, metadata={"help": "taco temperature"})
    tau_cgd: float = field(default=20, metadata={"help": "cgd temperature"})
    norm_grad: bool = field(default=False,
                            metadata={
                                "help": "use grad cosine similarity"})
    select_all: bool = field(default=True,
                             metadata={"help": "select all parameters?"})
    select_blocks: str = field(
        default=None,
        metadata={"help": "select blocks from the model"}
    )
    log_grads: bool = field(default=False,
                            metadata={"help": "log gradients"})
    log_gnorm: bool = field(
        default=False,
        metadata={"help": "log gnorm?"}
    )
    task_names: str = field(default='Marco,NQ,wow,trex,fever,zsre,aida',
                            metadata={"help": "task names"})
    weight_method: str = field(
        default='naive',
        metadata={"help": "weighting method for "
                          "multi_task training "
                          "(naive, taco, gn, pcg, cgd, uw) "})
    momentum_ipt: bool = field(default=True, metadata={"help": "momentum ipt?"})
    momentum_softmax: bool = field(default=False,
                                   metadata={"help": "momentum softmax ?"})
    norm_ipt: bool = field(default=True,
                           metadata={"help": "norm ipt or ipt exp avg?"})
    discourage: bool = field(default=False, metadata={
        "help": "discourage sensitive parameters?"})
    top_p: float = field(default=0.01,
                         metadata={"help": "top p for sampling ipts"})
    anneal: bool = field(default=False,
                         metadata={"help": "anneal softmax temperature?"})
    schedule_momentum: bool = field(default=False,
                                    metadata={"help": "use cosine momentum "
                                                      "scheduler?"})
    warmup_momentum: bool = field(default=False,
                                  metadata={"help": "warmup momentum "
                                                    "scheduler?"})
    softmax_dr: float = field(default=1e-3, metadata={
        "help": "softmax temperature decay rate"})
    save_block_indices: bool = field(default=False,
                                     metadata={
                                         "help": "save block sample indices?"})
    block_indices_path: str = field(default=None,
                                    metadata={"help": "block indices path "})
    block_sample: bool = field(default=False,
                               metadata={
                                   "help": "sample block-level ipts?"
                               })
    type_loss: Optional[str] = field(
        default="sum_log_nce",
        metadata={"help": "loss type for multi_label nce "
                          "(log_sum,sum_log,sum_log_nce,max_min)"}
    )
    multi_label: bool = field(
        default=False,
        metadata={"help": "multi_label retrieval or not"}
    )
    skip_steps_with_Nan: bool = field(
        default=False,
        metadata={"help": "whether to skip loss if nan"}
    )
    hard_negative_mining: bool = field(
        default=False,
        metadata={"help": "use standard hard negative mining strategy"
                          "(like hard-nce or ent-qa paper): keep "
                          "optimizer states instead of the decay lr strategy "
                          "in ANCE"}
    )
    epochs_per_hn: int = field(
        default=1,
        metadata={"help": "epochs per hard negative mining iteration"}
    )


@dataclass
class DenseEncodingArguments(DenseTrainingArguments):
    use_gpu: bool = field(default=False,
                          metadata={"help": "Use GPU for encoding"})
    split_retrieve: bool = field(
        default=False,
        metadata={"help":"split retrieve"}
    )
    encoded_save_path: str = field(default=None, metadata={
        "help": "where to save the encode"})
    trec_save_path: str = field(default=None, metadata={
        "help": "where to save the trec file"})
    topk: int = field(default=200, metadata={
        "help": "retrieve top k"
    })
    task_index: int = field(default=None, metadata={
        "help": "task index for using specific task head"
    })
    task_names: str = field(default='NQ,TQA,HOPO,WOW,TREX,FEVER,ZSRE,AIDA',
                            metadata={"help": "task names"})
    max_inmem_docs: int = field(default=3000000, metadata={
        "help": "max number of docs to keep in memory"})
