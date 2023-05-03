#!/bin/bash

HOME_DIR="/common/users/wz283/projects/"
CODE_DIR=$HOME_DIR"/TACO/"
TACO_DIR=$HOME_DIR"/taco_data/"
PLM_DIR=$TACO_DIR"/plm/"
MODEL_DIR=$TACO_DIR"/model/zeshel"
DATA_DIR=$TACO_DIR"/data/zeshel/"
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
if [ ! -d $ORIG_DIR ]; then
  echo "please download zeshel data from https://github.com/lajanugen/zeshel and put it in $ORIG_DIR"
else
  cd $CODE_DIR
  echo "preprocess data"
  python $CODE_DIR/scripts/zeshel/preprocess_data.py --input_dir $ORIG_DIR \
    --output_dir $RAW_DIR \
    --max_len 180
  echo "build train "
  python $CODE_DIR/scripts/zeshel/build_train.py \
        --tokenizer_name $PLM_DIR/t5-base-scaled  \
        --negative_file $ORIG_DIR/tfidf_candidates/train.json  \
        --qrels $RAW_DIR/train.qrel.tsv  \
        --queries $RAW_DIR/train.query.txt  \
        --collection $RAW_DIR/psg_corpus_train.tsv  \
        --save_to $PROCESSED_DIR/  \
        --template "Title: <title> Text: <text>" \
        --add_rand_negs \
        --num_rands 64 \
        --num_hards 64 \
        --truncate 128 \
        --use_doc_id_map
fi