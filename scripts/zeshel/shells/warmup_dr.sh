#!/bin/bash
HOME_DIR="/common/users/wz283/projects/"
CODE_DIR=$HOME_DIR"/TACO/"
TACO_DIR=$HOME_DIR"/taco_data/"
PLM_DIR=$TACO_DIR"/plm/"
MODEL_DIR=$TACO_DIR"/model/zeshel/warmup_dr/"
DATA_DIR=$TACO_DIR"/data/zeshel/"
RAW_DIR=$DATA_DIR"/raw/"
PROCESSED_DIR=$DATA_DIR"/processed/bm25/"
LOG_DIR=$TACO_DIR"/logs/zeshel/warmup_dr/"
EMBEDDING_DIR=$TACO_DIR"/embeddings/warmup_dr/"
RESULT_DIR=$TACO_DIR"/results/warmup_dr/"
EVAL_DIR=$TACO_DIR"/metrics/trec/trec_eval-9.0.7/trec_eval-9.0.7/"
ANCE_PROCESSED_DIR=$DATA_DIR"/processed/ance_dr/"
ANCE_MODEL_DIR=$TACO_DIR"/model/zeshel/ance_dr/"
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
epoch=4
lr=1e-5
p_len=160
log_step=100
bsz=16
n_passages=3
infer_bsz=1024
n_gpu=8
max_q_len=128

cd $CODE_DIR
export PYTHONPATH=.

echo "start warmup training of zeshel"

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
    --load_best_model_at_end True \
    --metric_for_best_model loss



echo "building dev index for zeshel"
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

echo "retrieve dev data of zeshel ... "
if [ ! -d "$RESULT_DIR/zeshel" ]; then
    mkdir -p $RESULT_DIR/zeshel
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
    --trec_save_path $RESULT_DIR/zeshel/dev.trec \
    --dataloader_num_workers 0

$EVAL_DIR/trec_eval -c -mrecip_rank.10 -mrecall.64,100 $RAW_DIR/dev.qrel.trec $RESULT_DIR/zeshel/dev.trec > $RESULT_DIR/zeshel/dev_results.txt
echo "building val hard negatives of ance first episode for zeshel ..."
mkdir -p $ANCE_PROCESSED_DIR/hn_iter_0
python src/taco/dataset/build_hn.py  \
    --tokenizer_name $PLM_DIR/t5-base-scaled  \
    --hn_file $RESULT_DIR/zeshel/dev.trec \
    --qrels $RAW_DIR/dev.qrel.tsv \
    --queries $RAW_DIR/dev.query.txt \
    --collection $RAW_DIR/psg_corpus_dev.tsv \
    --save_to $ANCE_PROCESSED_DIR/hn_iter_0 \
    --template "Title: <title> Text: <text>" \
    --add_rand_negs \
    --num_hards 64 \
    --num_rands 64 \
    --split dev \
    --seed 42 \
    --use_doc_id_map \
    --truncate $p_len

echo "splitting zeshel dev hn file"
tail -n 500 $ANCE_PROCESSED_DIR/hn_iter_0/dev_all.jsonl > $ANCE_PROCESSED_DIR/hn_iter_0/val.jsonl
#rm $ANCE_PROCESSED_DIR/hn_iter_0/dev_all.jsonl


echo "building test index for zeshel"
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

echo "retrieve test data of zeshel ... "

python -m src.taco.driver.retrieve  \
    --output_dir $EMBEDDING_DIR/ \
    --model_name_or_path $MODEL_DIR \
    --per_device_eval_batch_size $infer_bsz  \
    --query_path $RAW_DIR/test.query.txt  \
    --encoder_only False  \
    --query_template "<text>"  \
    --query_column_names  id,text \
    --q_max_len $max_q_len  \
    --fp16  \
    --trec_save_path $RESULT_DIR/zeshel/test.trec \
    --dataloader_num_workers 0

$EVAL_DIR/trec_eval -c -mrecip_rank.10 -mrecall.64,100 $RAW_DIR/test.qrel.trec $RESULT_DIR/zeshel/test.trec > $RESULT_DIR/zeshel/test_results.txt

echo "get preprocessed data of zeshel for ance training"
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
python -m src.taco.driver.retrieve  \
    --output_dir $EMBEDDING_DIR/ \
    --model_name_or_path $MODEL_DIR \
    --per_device_eval_batch_size $infer_bsz  \
    --query_path $RAW_DIR/train.query.txt  \
    --encoder_only False  \
    --query_template "<text>"  \
    --query_column_names  id,text \
    --q_max_len $max_q_len  \
    --fp16  \
    --trec_save_path $RESULT_DIR/zeshel/train.trec \
    --dataloader_num_workers 0 \
    --topk 100

echo "building hard negatives of ance first episode for zeshel ..."
mkdir -p $ANCE_PROCESSED_DIR/hn_iter_0
python src/taco/dataset/build_hn.py  \
    --tokenizer_name $PLM_DIR/t5-base-scaled  \
    --hn_file $RESULT_DIR/zeshel/train.trec \
    --qrels $RAW_DIR/train.qrel.tsv \
    --queries $RAW_DIR/train.query.txt \
    --collection $RAW_DIR/psg_corpus_train.tsv \
    --save_to $ANCE_PROCESSED_DIR/hn_iter_0 \
    --template "Title: <title> Text: <text>" \
    --add_rand_negs \
    --num_hards 64 \
    --num_rands 64 \
    --split train \
    --seed 42 \
    --use_doc_id_map \
    --truncate $p_len

echo "removing train zeshel trec files"
rm $RESULT_DIR/zeshel/train.trec

echo "splitting zeshel train hn file"

mv $ANCE_PROCESSED_DIR/hn_iter_0/train_all.jsonl > $ANCE_PROCESSED_DIR/hn_iter_0/train.jsonl


echo "moving warmed up model to ance iter 0 model folder for zeshel"
mv $MODEL_DIR  $ANCE_MODEL_DIR/hn_iter_0/
echo "deleting warmed up embeddings for zeshel"
rm $EMBEDDING_DIR/embeddings.corpus.rank.*


