#!/bin/bash
HOME_DIR="/common/users/wz283/projects/"
CACHE_DIR="/common/users/wz283/hf_dataset_cache/"
CODE_DIR=$HOME_DIR"/TACO/"
TACO_DIR=$HOME_DIR"/taco_data/"
PLM_DIR=$TACO_DIR"/plm/"
MODEL_DIR=$TACO_DIR"/model/ance_dr/msmarco/"
DATA_DIR=$TACO_DIR"/data/msmarco/"
WARM_MODEL_DIR=$TACO_DIR"/model/warmup_mt/mt_msmarco/"
RAW_DIR=$DATA_DIR"/raw/"
PROCESSED_DIR=$DATA_DIR"/processed/ance_dr/"
LOG_DIR=$TACO_DIR"/logs/msmarco/ance_dr/"
EMBEDDING_DIR=$TACO_DIR"/embeddings/ance_dr/"
RESULT_DIR=$TACO_DIR"/results/ance_dr/"
EVAL_DIR=$TACO_DIR"/metrics/trec/trec_eval-9.0.7/trec_eval-9.0.7/"
mkdir -p $TACO_DIR
mkdir -p $PLM_DIR
if [ -d $MODEL_DIR/hn_iter_0 ]; then
  echo "$MODEL_DIR/hn_iter_0 is not empty"
else
  echo "get initial model"
  mkdir -p $MODEL_DIR
  cp -r $WARM_MODEL_DIR  $MODEL_DIR/hn_iter_0
fi
mkdir -p $DATA_DIR
mkdir -p $RAW_DIR
mkdir -p $PROCESSED_DIR
mkdir -p $LOG_DIR
mkdir -p $EMBEDDING_DIR
mkdir -p $RESULT_DIR
mkdir -p $EVAL_DIR
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

eval_delay=0
epoch=6
p_len=160
max_q_len=128
log_step=100
n_passages=8
rands_ratio=0.5
num_hn_iters=6
epoch_per_hn=1
lr=5e-6
dr=0.8
n_gpu=8
bsz=24
infer_bsz=4096
steps=250
n_gpu=8
iter_num=-1
for ((hn_iter=0; hn_iter<$num_hn_iters; hn_iter++))
do
    echo "Iteration $hn_iter"
    let new_hn_iter=$hn_iter+1
    let iter_num=$iter_num+1
    if [ ${hn_iter} == 0 ]; then
      echo "initial processed data should be obtained after warmup training"
      if [ -d $PROCESSED_DIR/hn_iter_0 ]; then
        echo "initial processed data already exists"
      else
        echo "copy from naive processed data"
        NAIVE_INIT_DIR=$DATA_DIR/processed/ance_hn_mt/mt_msmarco/naive/hn_iter_0
        cp -r $NAIVE_INIT_DIR $PROCESSED_DIR/hn_iter_0
      fi
    else
      echo "retrieving train ..."
      export RANDOM=$hn_iter
      echo "random down_sample ... "
      shuf -n 100000 $RAW_DIR/train.query.txt > $PROCESSED_DIR/hn_iter_${hn_iter}/train.query.txt
      mkdir -p $RESULT_DIR/msmarco/hn_iter_${hn_iter}
      python -m src.taco.driver.retrieve  \
          --output_dir $EMBEDDING_DIR/ \
          --model_name_or_path $MODEL_DIR/hn_iter_${hn_iter} \
          --per_device_eval_batch_size $infer_bsz  \
          --query_path $PROCESSED_DIR/hn_iter_${hn_iter}/train.query.txt  \
          --encoder_only False  \
          --query_template "<text>"  \
          --query_column_names  id,text \
          --q_max_len $max_q_len  \
          --fp16  \
          --trec_save_path $RESULT_DIR/msmarco/hn_iter_${hn_iter}/train.trec \
          --dataloader_num_workers 32 \
          --topk 100 \
          --cache_dir $CACHE_DIR \
          --split_retrieve \
          --use_gpu
      echo "building hard negatives for msmarco ..."
      mkdir -p $PROCESSED_DIR/hn_iter_${hn_iter}
      python src/taco/dataset/build_hn.py  \
          --tokenizer_name $PLM_DIR/t5-base-scaled  \
          --hn_file $RESULT_DIR/msmarco/hn_iter_${hn_iter}/train.trec \
          --qrels $RAW_DIR/train.qrel.tsv \
          --queries $PROCESSED_DIR/hn_iter_${hn_iter}/train.query.txt \
          --collection $RAW_DIR/psg_corpus.tsv \
          --save_to $PROCESSED_DIR/hn_iter_${hn_iter} \
          --template "Title: <title> Text: <text>" \
          --num_hards 32 \
          --num_rands 32 \
          --split train \
          --seed ${hn_iter} \
          --truncate $p_len \
          --cache_dir $CACHE_DIR

      echo "removing training trec file of msmarco"
      rm $RESULT_DIR/msmarco/hn_iter_${hn_iter}/train.trec

      echo "splitting msmarco train hn file"
      tail -n 500 $PROCESSED_DIR/hn_iter_${hn_iter}/train_all.jsonl > $PROCESSED_DIR/hn_iter_${hn_iter}/val.jsonl
      head -n -500 $PROCESSED_DIR/hn_iter_${hn_iter}/train_all.jsonl > $PROCESSED_DIR/hn_iter_${hn_iter}/train.jsonl
      rm $PROCESSED_DIR/hn_iter_${hn_iter}/train_all.jsonl
    fi


    echo "start hn training for msmarco for episode-${hn_iter} ..."
#    if [ $hn_iter != 0 ]; then
#      resume=$(ls -td $MODEL_DIR/checkpoint-* | head -1)
#    else
#      resume=False
#    fi
    torchrun --nproc_per_node=$n_gpu --standalone --nnodes=1 src/taco/driver/train_dr.py \
        --output_dir $MODEL_DIR/hn_iter_${new_hn_iter}  \
        --model_name_or_path $MODEL_DIR/hn_iter_${hn_iter}  \
        --do_train  \
        --eval_delay $eval_delay \
        --save_strategy epoch \
        --evaluation_strategy epoch \
        --logging_steps $log_step \
        --train_path $PROCESSED_DIR/hn_iter_${hn_iter}/train.jsonl  \
        --eval_path $PROCESSED_DIR/hn_iter_${hn_iter}/val.jsonl \
        --fp16  \
        --per_device_train_batch_size $bsz  \
        --train_n_passages $n_passages  \
        --learning_rate $lr  \
        --q_max_len $max_q_len  \
        --p_max_len $p_len \
        --num_train_epochs $epoch_per_hn  \
        --epochs_per_hn $epoch_per_hn \
        --logging_dir $LOG_DIR/hn_iter_${hn_iter}  \
        --negatives_x_device True \
        --remove_unused_columns False \
        --overwrite_output_dir True \
        --dataloader_num_workers 0 \
        --multi_label False \
        --in_batch_negatives True \
        --pooling first \
        --positive_passage_no_shuffle True \
        --negative_passage_no_shuffle True \
        --add_rand_negs False \
        --encoder_only False \
        --save_total_limit 2 \
        --load_best_model_at_end False \
        --metric_for_best_model loss \
        --hard_negative_mining False \
        --rands_ratio $rands_ratio \
        --data_cache_dir $CACHE_DIR \
        --total_iter_num $num_hn_iters \
        --iter_num $hn_iter
#        --resume_from_checkpoint $resume \

    echo "evaluating msmarco dev for episode-${hn_iter} ..."
    echo "building index for msmarco  for episode-${hn_iter} "
#    torchrun --nproc_per_node=$n_gpu --standalone --nnodes=1 src/taco/driver/build_index.py \
    python src/taco/driver/build_index.py  \
      --output_dir $EMBEDDING_DIR/ \
      --model_name_or_path $MODEL_DIR/hn_iter_${new_hn_iter} \
      --per_device_eval_batch_size $infer_bsz  \
      --corpus_path $RAW_DIR/psg_corpus.tsv  \
      --encoder_only False  \
      --doc_template "Title: <title> Text: <text>"  \
      --doc_column_names id,title,text \
      --q_max_len $max_q_len  \
      --p_max_len $p_len  \
      --fp16  \
      --dataloader_num_workers 32 \
      --cache_dir $CACHE_DIR

    echo "retrieve dev data of msmarco for episode-${hn_iter} ... "
    if [ ! -d "$RESULT_DIR/msmarco/hn_iter_${new_hn_iter}" ]; then
        mkdir -p $RESULT_DIR/msmarco/hn_iter_${new_hn_iter}
    fi

    python -m src.taco.driver.retrieve  \
        --output_dir $EMBEDDING_DIR/ \
        --model_name_or_path $MODEL_DIR/hn_iter_${new_hn_iter} \
        --per_device_eval_batch_size $infer_bsz  \
        --query_path $RAW_DIR/dev.query.txt  \
        --encoder_only False  \
        --query_template "<text>"  \
        --query_column_names  id,text \
        --q_max_len $max_q_len  \
        --fp16  \
        --trec_save_path $RESULT_DIR/msmarco/hn_iter_${new_hn_iter}/dev.trec \
        --dataloader_num_workers 32 \
        --topk 100 \
        --cache_dir $CACHE_DIR \
        --split_retrieve \
        --use_gpu

    $EVAL_DIR/trec_eval -c -mrecip_rank.10 -mrecall.64,100 $RAW_DIR/dev.qrel.trec $RESULT_DIR/msmarco/hn_iter_${new_hn_iter}/dev.trec > $RESULT_DIR/msmarco/hn_iter_${new_hn_iter}/dev_results.txt
done
echo "deleting embedding cache"
rm $EMBEDDING_DIR/embeddings.corpus.rank.*
