#!/bin/bash
HOME_DIR="/common/users/wz283/projects/"
CACHE_DIR="/common/users/wz283/hf_dataset_cache/"
CODE_DIR=$HOME_DIR"/TACO/"
TACO_DIR=$HOME_DIR"/taco_data/"
PLM_DIR=$TACO_DIR"/plm/"
MODEL_DIR=$TACO_DIR"/model/msmarco/warmup_nce_dr/"
DATA_DIR=$TACO_DIR"/data/msmarco/"
RAW_DIR=$DATA_DIR"/raw/"
PROCESSED_DIR=$DATA_DIR"/processed/bm25/"
LOG_DIR=$TACO_DIR"/logs/msmarco/warmup_nce_dr/"
EMBEDDING_DIR=$TACO_DIR"/embeddings/warmup_nce_dr/"
RESULT_DIR=$TACO_DIR"/results/warmup_nce_dr/"
EVAL_DIR=$TACO_DIR"/metrics/trec/trec_eval-9.0.7/trec_eval-9.0.7/"
ANCE_PROCESSED_DIR=$DATA_DIR"/processed/hard_nce_dr/"
ANCE_MODEL_DIR=$TACO_DIR"/model/msmarco/hard_nce_dr/"
mkdir -p $TACO_DIR
mkdir -p $PLM_DIR
mkdir -p $MODEL_DIR
mkdir -p $DATA_DIR
mkdir -p $RAW_DIR
mkdir -p $PROCESSED_DIR
mkdir -p $LOG_DIR
mkdir -p $EMBEDDING_DIR
mkdir -p $RESULT_DIR
mkdir -p $EVAL_DIR
mkdir -p $ANCE_MODEL_DIR
mkdir -p $ANCE_PROCESSED_DIR
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

SAVE_STEP=10000
EVAL_STEP=300

eval_delay=0
epoch=10
lr=1e-5
p_len=160
log_step=100
bsz=2
n_passages=64
infer_bsz=1024
n_gpu=8
max_q_len=128

cd $CODE_DIR
export PYTHONPATH=.

echo "start warmup training of msmarco"

torchrun --nproc_per_node=$n_gpu --standalone --nnodes=1 src/taco/driver/train_dr.py \
    --output_dir $MODEL_DIR  \
    --model_name_or_path $PLM_DIR/t5-base-scaled  \
    --do_train  \
    --eval_delay $eval_delay \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --logging_steps $log_step \
    --train_path $PROCESSED_DIR/train.jsonl  \
    --eval_path $PROCESSED_DIR/val.jsonl \
    --fp16  \
    --per_device_train_batch_size $bsz  \
    --train_n_passages $n_passages  \
    --learning_rate $lr  \
    --q_max_len $max_q_len  \
    --p_max_len $p_len \
    --num_train_epochs $epoch  \
    --logging_dir $LOG_DIR  \
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
    --load_best_model_at_end True \
    --metric_for_best_model loss \
    --data_cache_dir $CACHE_DIR



echo "building dev index for msmarco"
#  python src/taco/driver/build_index.py  \
torchrun --nproc_per_node=$n_gpu --standalone --nnodes=1 src/taco/driver/build_index.py \
    --output_dir $EMBEDDING_DIR/ \
    --model_name_or_path $MODEL_DIR \
    --per_device_eval_batch_size $infer_bsz  \
    --corpus_path $RAW_DIR/psg_corpus.tsv  \
    --encoder_only False  \
    --doc_template "Title: <title> Text: <text>"  \
    --doc_column_names id,title,text \
    --q_max_len $max_q_len  \
    --p_max_len $p_len  \
    --fp16  \
    --dataloader_num_workers 0

echo "retrieve dev data of msmarco ... "
if [ ! -d "$RESULT_DIR/msmarco" ]; then
    mkdir -p $RESULT_DIR/msmarco
fi

python -m src.taco.driver.retrieve  \
    --output_dir $EMBEDDING_DIR/ \
    --model_name_or_path $MODEL_DIR \
    --per_device_eval_batch_size $infer_bsz  \
    --query_path $RAW_DIR/dev.query.txt  \
    --encoder_only False  \
    --query_template "<text>"  \
    --query_column_names  id,text \
    --q_max_len $max_q_len  \
    --fp16  \
    --trec_save_path $RESULT_DIR/msmarco/dev.trec \
    --dataloader_num_workers 0

$EVAL_DIR/trec_eval -c -mrecip_rank.10 -mrecall.64,100 $RAW_DIR/dev.qrel.trec $RESULT_DIR/msmarco/dev.trec > $RESULT_DIR/msmarco/dev_results.txt

echo "get preprocessed data of msmarco for ance training"
#  python src/taco/driver/build_index.py  \


export RANDOM=42
echo "random down_sample train queries ... "
shuf -n 100000 $RAW_DIR/train.query.txt > $ANCE_PROCESSED_DIR/hn_iter_0/train.query.txt
echo "retrieving train ..."
python -m src.taco.driver.retrieve  \
    --output_dir $EMBEDDING_DIR/ \
    --model_name_or_path $MODEL_DIR \
    --per_device_eval_batch_size $infer_bsz  \
    --query_path $ANCE_PROCESSED_DIR/hn_iter_0/train.query.txt  \
    --encoder_only False  \
    --query_template "<text>"  \
    --query_column_names  id,text \
    --q_max_len $max_q_len  \
    --fp16  \
    --trec_save_path $RESULT_DIR/msmarco/train.trec \
    --dataloader_num_workers 0 \
    --topk 100

echo "building hard negatives of ance first episode for msmarco ..."
mkdir -p $ANCE_PROCESSED_DIR/hn_iter_0
python src/taco/dataset/build_hn.py  \
    --tokenizer_name $PLM_DIR/t5-base-scaled  \
    --hn_file $RESULT_DIR/msmarco/train.trec \
    --qrels $RAW_DIR/train.qrel.tsv \
    --queries $ANCE_PROCESSED_DIR/hn_iter_0/train.query.txt \
    --collection $RAW_DIR/psg_corpus.tsv \
    --save_to $ANCE_PROCESSED_DIR/hn_iter_0 \
    --template "Title: <title> Text: <text>" \
    --num_hards 64 \
    --num_rands 64 \
    --split train \
    --seed 42 \
    --add_rand_negs \
    --truncate $p_len

echo "removing train msmarco trec files"
rm $RESULT_DIR/msmarco/train.trec

echo "splitting msmarco train hn file"

tail -n 500 $ANCE_PROCESSED_DIR/hn_iter_0/train_all.jsonl > $ANCE_PROCESSED_DIR/hn_iter_0/val.jsonl
head -n -500 $ANCE_PROCESSED_DIR/hn_iter_0/train_all.jsonl > $ANCE_PROCESSED_DIR/hn_iter_0/train.jsonl
rm $ANCE_PROCESSED_DIR/hn_iter_0/train_all.jsonl

echo "remove checkpoints"
rm $MODEL_DIR/checkpoint-*
echo "copy warmed up model to ance iter 0 model folder for msmarco"
cp -r $MODEL_DIR  $ANCE_MODEL_DIR/hn_iter_0/
echo "deleting warmed up embeddings for msmarco"
rm $EMBEDDING_DIR/embeddings.corpus.rank.*


