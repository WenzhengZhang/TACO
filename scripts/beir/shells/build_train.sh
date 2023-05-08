#!/bin/bash
DATA_NAME=$1
HOME_DIR="/common/users/wz283/projects/"
CODE_DIR=$HOME_DIR"/TACO/"
TACO_DIR=$HOME_DIR"/taco_data/"
PLM_DIR=$TACO_DIR"/plm/"
MODEL_DIR=$TACO_DIR"/model/beir"
DATA_DIR=$TACO_DIR"/data/beir/"
ORIG_DIR=$DATA_DIR"/original/"
#RAW_DIR=$DATA_DIR"/raw/"
#PROCESSED_DIR=$DATA_DIR"/processed/bm25/"
mkdir -p $TACO_DIR
mkdir -p $PLM_DIR
mkdir -p $MODEL_DIR
mkdir -p $DATA_DIR
#mkdir -p $RAW_DIR
#mkdir -p $PROCESSED_DIR
mkdir -p $EVAL_DIR
mkdir -p $ORIG_DIR
cd $CODE_DIR

#echo "get train bm25 candidates first ... "
#if [ ${DATA_NAME} == nq ]; then
#  echo "build lucene index"
#  python -m pyserini.index.lucene \
#    --collection BeirFlatCollection \
#    --input $ORIG_DIR/${DATA_NAME}/ \
#    --index $ORIG_DIR/indexes/beir-nq-train \
#    --generator DefaultLuceneDocumentGenerator \
#    --threads 16 \
#    --storePositions --storeDocvectors --storeRaw --optimize
#    echo "bm25 retrieve for nq train"
#    python -m pyserini.search.lucene   \
#      --index $ORIG_DIR/indexes/beir-nq-train   \
#      --topics $DATA_DIR/nq/raw/train.query.txt \
#      --output $DATA_DIR/nq/raw/train.bm25.txt   \
#      --output-format trec   \
#      --batch 36 --threads 12 \
#      --hits 100 \
#      --bm25 \
#      --remove-query
#else
#  python -m pyserini.search.lucene   \
#    --index beir-v1.0.0-${DATA_NAME}-flat   \
#    --topics $DATA_DIR/${DATA_NAME}/raw/train.query.txt \
#    --output $DATA_DIR/${DATA_NAME}/raw/train.bm25.txt   \
#    --output-format trec   \
#    --batch 36 --threads 12 \
#    --hits 100 \
#    --bm25 \
#    --remove-query
#fi

echo "build training data for warmup training ... "
p_len=160
mkdir -p $DATA_DIR/${DATA_NAME}/processed/bm25/
if [ ${DATA_NAME} == nq ]; then
  corpus_path=$DATA_DIR/${DATA_NAME}/raw/psg_corpus_train.tsv
else
  corpus_path=$DATA_DIR/${DATA_NAME}/raw/psg_corpus.tsv
fi
python src/taco/dataset/build_hn.py  \
  --tokenizer_name $PLM_DIR/t5-base-scaled  \
  --hn_file $DATA_DIR/${DATA_NAME}/raw/train.bm25.txt \
  --qrels $DATA_DIR/${DATA_NAME}/raw/train.qrel.tsv \
  --queries $DATA_DIR/${DATA_NAME}/raw/train.query.txt \
  --collection $corpus_path \
  --save_to $DATA_DIR/${DATA_NAME}/processed/bm25/ \
  --template "Title: <title> Text: <text>" \
  --add_rand_negs \
  --num_hards 32 \
  --num_rands 32 \
  --split train \
  --seed 42 \
  --truncate $p_len
#  --use_doc_id_map

echo "split train into train and val"
tail -n 500 $DATA_DIR/${DATA_NAME}/processed/bm25/train_all.jsonl > $DATA_DIR/${DATA_NAME}/processed/bm25/val.jsonl
head -n -500 $DATA_DIR/${DATA_NAME}/processed/bm25/train_all.jsonl > $DATA_DIR/${DATA_NAME}/processed/bm25/train.jsonl
rm $DATA_DIR/${DATA_NAME}/processed/bm25/train_all.jsonl
