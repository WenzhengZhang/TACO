#!/bin/bash
DATA_NAME=$1
HOME_DIR="/common/users/wz283/projects/"
CODE_DIR=$HOME_DIR"/TACO/"
TACO_DIR=$HOME_DIR"/taco_data/"
PLM_DIR=$TACO_DIR"/plm/"
MODEL_DIR=$TACO_DIR"/model/beir"
DATA_DIR=$TACO_DIR"/data/beir/"
ORIG_DIR=$DATA_DIR"/original/"
RAW_DIR=$DATA_DIR"/raw/"
PROCESSED_DIR=$DATA_DIR"/processed/bm25/"
mkdir -p $TACO_DIR
mkdir -p $PLM_DIR
mkdir -p $MODEL_DIR
mkdir -p $DATA_DIR
mkdir -p $RAW_DIR
mkdir -p $PROCESSED_DIR
mkdir -p $EVAL_DIR
mkdir -p $ORIG_DIR
cd $ORIG_DIR

#echo "get train bm25 candidates first ... "
#python -m pyserini.search.lucene   \
#  --index beir-v1.0.0-${DATA_NAME}-flat   \
#  --topics $RAW_DIR/${DATA_NAME}/train.query.txt \
#  --output $RAW_DIR/${DATA_NAME}/train.bm25.txt   \
#  --output-format trec   \
#  --batch 36 --threads 12 \
#  --hits 100 \
#  --bm25 \
#  --remove-query

echo "build training data for warmup training ... "
p_len=160
mkdir -p $PROCESSED_DIR/${DATA_NAME}/
python src/taco/dataset/build_hn.py  \
  --tokenizer_name $PLM_DIR/t5-base-scaled  \
  --hn_file $RAW_DIR/${DATA_NAME}/train.bm25.txt \
  --qrels $RAW_DIR/${DATA_NAME}/train.qrel.tsv \
  --queries $RAW_DIR/${DATA_NAME}/train.query.txt \
  --collection $RAW_DIR/${DATA_NAME}/psg_corpus.tsv \
  --save_to $PROCESSED_DIR/${DATA_NAME}/ \
  --template "Title: <title> Text: <text>" \
  --add_rand_negs \
  --num_hards 48 \
  --num_rands 48 \
  --split train \
  --seed 42 \
  --truncate $p_len \
  --use_doc_id_map
