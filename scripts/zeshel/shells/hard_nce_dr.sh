#!/bin/bash
HOME_DIR="/common/users/wz283/projects/"
CODE_DIR=$HOME_DIR"/TACO/"
TACO_DIR=$HOME_DIR"/taco_data/"
PLM_DIR=$TACO_DIR"/plm/"
MODEL_DIR=$TACO_DIR"/model/zeshel/hard_nce_dr/"
DATA_DIR=$TACO_DIR"/data/zeshel/"
RAW_DIR=$DATA_DIR"/raw/"
PROCESSED_DIR=$DATA_DIR"/processed/hard_nce_dr/"
LOG_DIR=$TACO_DIR"/logs/zeshel/hard_nce_dr/"
EMBEDDING_DIR=$TACO_DIR"/embeddings/hard_nce_dr/"
RESULT_DIR=$TACO_DIR"/results/hard_nce_dr/"
EVAL_DIR=$TACO_DIR"/metrics/trec/trec_eval-9.0.7/trec_eval-9.0.7/"
mkdir -p $TACO_DIR
mkdir -p $PLM_DIR
if [ -d $MODEL_DIR ]; then
  echo "$MODEL_DIR is not empty"
else
  echo "get initial model"
  cp -r $PLM_DIR/t5-base-scaled/ $MODEL_DIR
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
epoch=4
p_len=160
max_q_len=128
log_step=100
n_passages=64
rands_ratio=0.5
num_hn_iters=4
epoch_per_hn=1
lr=1e-5
dr=1
n_gpu=8
bsz=2
infer_bsz=256
steps=250
n_gpu=8

for ((hn_iter=0; hn_iter<$num_hn_iters; hn_iters++))
do
    echo "Iteration $hn_iter"
    let new_hn_iter=$hn_iter+1
    if [ $hn_iter != 0 ]; then
      echo "build dev index and retrieve for the first episode"
      echo "build index ... "
      torchrun --nproc_per_node=$n_gpu --standalone --nnodes=1 src/taco/driver/build_index.py \
        --output_dir $EMBEDDING_DIR/ \
        --model_name_or_path $MODEL_DIR \
        --per_device_eval_batch_size $infer_bsz  \
        --corpus_path $RAW_DIR/psg_corpus_dev.tsv  \
        --encoder_only False  \
        --doc_template "Title: <title> Text: <text>"  \
        --doc_column_names id,title,text \
        --q_max_len $max_q_len  \
        --p_max_len $p_len  \
        --fp16  \
        --dataloader_num_workers 0
      echo "retrieve ... "
      python -m src.taco.driver.retrieve  \
        --output_dir $EMBEDDING_DIR/ \
        --model_name_or_path $MODEL_DIR \
        --per_device_eval_batch_size $infer_bsz  \
        --query_path $RAW_DIR/zeshel/dev.query.txt  \
        --encoder_only False  \
        --query_template "<text>"  \
        --query_column_names  id,text \
        --q_max_len $max_q_len  \
        --fp16  \
        --trec_save_path $RESULT_DIR/zeshel/hn_iter_${hn_iter}/dev.trec \
        --dataloader_num_workers 0
    fi
    echo "building val hard negatives for zeshel ..."
    mkdir -p $PROCESSED_DIR/hn_iter_${hn_iter}
    python src/taco/dataset/build_hn.py  \
        --tokenizer_name $PLM_DIR/t5-base-scaled  \
        --hn_file $RESULT_DIR/zeshel/hn_iter_${hn_iter}/dev.trec \
        --qrels $RAW_DIR/zeshel/dev.qrel.tsv \
        --queries $RAW_DIR/zeshel/dev.query.txt \
        --collection $RAW_DIR/psg_corpus_dev.tsv \
        --save_to $PROCESSED_DIR/hn_iter_${hn_iter} \
        --template "Title: <title> Text: <text>" \
        --add_rand_negs \
        --num_hards 64 \
        --num_rands 64 \
        --split dev \
        --seed ${hn_iter} \
        --use_doc_id_map \
        --truncate $p_len
    echo "splitting zeshel dev hn file"
    tail -n 500 $PROCESSED_DIR/hn_iter_${hn_iter}/dev_all.jsonl > $PROCESSED_DIR/hn_iter_${hn_iter}/val.jsonl
    echo "building train index for zeshel"
    #  python src/taco/driver/build_index.py  \
    torchrun --nproc_per_node=$n_gpu --standalone --nnodes=1 src/taco/driver/build_index.py \
        --output_dir $EMBEDDING_DIR/ \
        --model_name_or_path $MODEL_DIR \
        --per_device_eval_batch_size $infer_bsz  \
        --corpus_path $RAW_DIR/psg_corpus_train.tsv  \
        --encoder_only False  \
        --doc_template "Title: <title> Text: <text>"  \
        --doc_column_names id,title,text \
        --q_max_len $max_q_len  \
        --p_max_len $p_len  \
        --fp16  \
        --dataloader_num_workers 0
    echo "retrieving train ..."
    mkdir -p $RESULT_DIR/zeshel/hn_iter_${hn_iter}
    python -m src.taco.driver.retrieve  \
        --output_dir $EMBEDDING_DIR/ \
        --model_name_or_path $MODEL_DIR \
        --per_device_eval_batch_size $infer_bsz  \
        --query_path $RAW_DIR/zeshel/train.query.txt  \
        --encoder_only False  \
        --query_template "<text>"  \
        --query_column_names  id,text \
        --q_max_len $max_q_len  \
        --fp16  \
        --trec_save_path $RESULT_DIR/zeshel/hn_iter_${hn_iter}/train.trec \
        --dataloader_num_workers 0 \
        --topk 100
    echo "building hard negatives for zeshel ..."
    mkdir -p $PROCESSED_DIR/hn_iter_${hn_iter}
    python src/taco/dataset/build_hn.py  \
        --tokenizer_name $PLM_DIR/t5-base-scaled  \
        --hn_file $RESULT_DIR/zeshel/hn_iter_${hn_iter}/train.trec \
        --qrels $RAW_DIR/zeshel/train.qrel.tsv \
        --queries $RAW_DIR/zeshel/train.query.txt \
        --collection $RAW_DIR/psg_corpus.tsv \
        --save_to $PROCESSED_DIR/hn_iter_${hn_iter} \
        --template "Title: <title> Text: <text>" \
        --add_rand_negs \
        --num_hards 64 \
        --num_rands 64 \
        --split train \
        --seed ${hn_iter} \
        --use_doc_id_map \
        --truncate $p_len

    echo "removing training trec file of zeshel"
    rm $RESULT_DIR/zeshel/hn_iter_${hn_iter}/train.trec

    echo "splitting zeshel train hn file"
    mv $PROCESSED_DIR/hn_iter_0/train_all.jsonl > $PROCESSED_DIR/hn_iter_${hn_iter}/train.jsonl


    echo "start hn training for zeshel for episode-${hn_iter} ..."
    if [ $hn_iter != 0 ]; then
      resume=$(ls -td $MODEL_DIR/checkpoint-* | head -1)
    else
      resume=False
    fi
    torchrun --nproc_per_node=$n_gpu --standalone --nnodes=1 src/taco/driver/train_dr.py \
        --output_dir $MODEL_DIR  \
        --model_name_or_path $MODEL_DIR  \
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
        --num_train_epochs $epoch  \
        --epochs_per_hn $epoch_per_hn \
        --logging_dir $LOG_DIR/hn_iter_${hn_iter}  \
        --negatives_x_device False \
        --remove_unused_columns False \
        --overwrite_output_dir True \
        --dataloader_num_workers 0 \
        --multi_label False \
        --in_batch_negatives False \
        --pooling first \
        --positive_passage_no_shuffle True \
        --negative_passage_no_shuffle True \
        --add_rand_negs True \
        --encoder_only False \
        --save_total_limit 2 \
        --load_best_model_at_end False \
        --metric_for_best_model loss \
        --hard_negative_mining True \
        --rands_ratio $rands_ratio \
        --resume_from_checkpoint $resume

    echo "evaluating zeshel dev for episode-${hn_iter} ..."
    echo "building index for zeshel dev for episode-${hn_iter} "
    #  python src/taco/driver/build_index.py  \
    torchrun --nproc_per_node=$n_gpu --standalone --nnodes=1 src/taco/driver/build_index.py \
        --output_dir $EMBEDDING_DIR/ \
        --model_name_or_path $MODEL_DIR \
        --per_device_eval_batch_size $infer_bsz  \
        --corpus_path $RAW_DIR/psg_corpus_dev.tsv  \
        --encoder_only False  \
        --doc_template "Title: <title> Text: <text>"  \
        --doc_column_names id,title,text \
        --q_max_len $max_q_len  \
        --p_max_len $p_len  \
        --fp16  \
        --dataloader_num_workers 0

    echo "retrieve dev data of zeshel for episode-${hn_iter} ... "
    if [ ! -d "$RESULT_DIR/zeshel/hn_iter_${new_hn_iter}" ]; then
        mkdir -p $RESULT_DIR/zeshel/hn_iter_${new_hn_iter}
    fi

    python -m src.taco.driver.retrieve  \
        --output_dir $EMBEDDING_DIR/ \
        --model_name_or_path $MODEL_DIR \
        --per_device_eval_batch_size $infer_bsz  \
        --query_path $RAW_DIR/zeshel/dev.query.txt  \
        --encoder_only False  \
        --query_template "<text>"  \
        --query_column_names  id,text \
        --q_max_len $max_q_len  \
        --fp16  \
        --trec_save_path $RESULT_DIR/zeshel/hn_iter_${new_hn_iter}/dev.trec \
        --dataloader_num_workers 0

    $EVAL_DIR/trec_eval -c -mrecip_rank.10 -mrecall.64,100 $RAW_DIR/zeshel/dev.qrel.trec $RESULT_DIR/zeshel/hn_iter_${new_hn_iter}/dev.trec > $RESULT_DIR/zeshel/hn_iter_${new_hn_iter}/dev_results.txt

    echo "evaluating zeshel test for episode-${hn_iter} ..."
    echo "building index for zeshel test for episode-${hn_iter} "
    #  python src/taco/driver/build_index.py  \
    torchrun --nproc_per_node=$n_gpu --standalone --nnodes=1 src/taco/driver/build_index.py \
        --output_dir $EMBEDDING_DIR/ \
        --model_name_or_path $MODEL_DIR \
        --per_device_eval_batch_size $infer_bsz  \
        --corpus_path $RAW_DIR/psg_corpus_test.tsv  \
        --encoder_only False  \
        --doc_template "Title: <title> Text: <text>"  \
        --doc_column_names id,title,text \
        --q_max_len $max_q_len  \
        --p_max_len $p_len  \
        --fp16  \
        --dataloader_num_workers 0

    echo "retrieve test data of zeshel for episode-${hn_iter} ... "
    if [ ! -d "$RESULT_DIR/zeshel/hn_iter_${new_hn_iter}" ]; then
        mkdir -p $RESULT_DIR/zeshel/hn_iter_${new_hn_iter}
    fi

    python -m src.taco.driver.retrieve  \
        --output_dir $EMBEDDING_DIR/ \
        --model_name_or_path $MODEL_DIR \
        --per_device_eval_batch_size $infer_bsz  \
        --query_path $RAW_DIR/zeshel/test.query.txt  \
        --encoder_only False  \
        --query_template "<text>"  \
        --query_column_names  id,text \
        --q_max_len $max_q_len  \
        --fp16  \
        --trec_save_path $RESULT_DIR/zeshel/hn_iter_${new_hn_iter}/test.trec \
        --dataloader_num_workers 0

    $EVAL_DIR/trec_eval -c -mrecip_rank.10 -mrecall.64,100 $RAW_DIR/zeshel/test.qrel.trec $RESULT_DIR/zeshel/hn_iter_${new_hn_iter}/test.trec > $RESULT_DIR/zeshel/hn_iter_${new_hn_iter}/test_results.txt
done
echo "deleting embedding cache"
rm $EMBEDDING_DIR/embeddings.corpus.rank.*
