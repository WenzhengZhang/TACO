#!/bin/bash

HOME_DIR="/common/users/wz283/projects/"
CODE_DIR=$HOME_DIR"/TACO/"
TACO_DIR=$HOME_DIR"/taco_data/"
PLM_DIR=$TACO_DIR"/plm/"
MODEL_DIR=$TACO_DIR"/model/msmarco"
DATA_DIR=$TACO_DIR"/data/msmarco/"
RAW_DIR=$DATA_DIR"/raw/"
PROCESSED_DIR=$DATA_DIR"/processed/bm25/"
EVAL_DIR=$TACO_DIR"/metrics/trec/trec_eval-9.0.7/"
ORIG_DIR=$DATA_DIR"/original/"
mkdir -p $TACO_DIR
mkdir -p $PLM_DIR
mkdir -p $MODEL_DIR
mkdir -p $DATA_DIR
mkdir -p $RAW_DIR
mkdir -p $PROCESSED_DIR
mkdir -p $EVAL_DIR
mkdir -p $PROCESSED_DIR
mkdir -p $ORIG_DIR


### download pretrained language model, in this case T5
#if [ -d "$PLM_DIR/t5-base-scaled" ]; then
#    echo "$PLM_DIR/t5-base-scaled already exists.";
#else
#    echo "downloading T5 checkpoint into $PLM_DIR/t5-base-scaled";
#    python lib/scripts/scale_t5_weights.py \
#        --input_model_path $PLM_DIR/t5-base \
#        --output_model_path $PLM_DIR/t5-base-scaled \
#        --model_name_or_path t5-base\
#        --num_layers 12
#
#fi
#
#### download and process data
#cd $ORIG_DIR
#
#### if you want titles of msmarco documents from these authors
#wget --no-check-certificate https://rocketqa.bj.bcebos.com/corpus/marco.tar.gz
#tar -zxf marco.tar.gz
#rm -rf marco.tar.gz
#mv marco rocketQA-marco
#
#### move title and passages from RocketQA to different folder
#cp rocketQA-marco/para.txt $ORIG_DIR
#cp rocketQA-marco/para.title.txt $ORIG_DIR
#cp rocketQA-marco/train.query.txt $ORIG_DIR
#
#### work in this folder for now
#cd $ORIG_DIR
#
#if [ ! -f "$ORIG_DIR/collection.tsv" ]; then
#    wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
#    tar -zxvf collectionandqueries.tar.gz
#    rm collectionandqueries.tar.gz
#
#fi
#
#if [ ! -f "$ORIG_DIR/triples.train.small.tsv" ]; then
#    wget https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.small.tar.gz
#    tar -zxvf triples.train.small.tar.gz
#    rm triples.train.small.tar.gz*
#
#fi
#
#
#if [ ! -f "$ORIG_DIR/qrels.train.tsv" ]; then
#    wget https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv -O qrels.train.tsv
#
#fi
#
#
#if [ ! -f "$ORIG_DIR/qidpidtriples.train.full.2.tsv" ]; then
#    wget https://msmarco.blob.core.windows.net/msmarcoranking/qidpidtriples.train.full.2.tsv.gz
#    gunzip qidpidtriples.train.full.2.tsv.gz
#
#fi
#
#
#### if you want to join titles with the msmarco passages
#if [ -f "$ORIG_DIR/collection_with_title.tsv" ]; then
#    echo "$ORIG_DIR/collection_with_title.tsv exists.";
#else
#    echo "Joining para.txt and para.title.txt";
#    join  -t "$(echo -en '\t')"  -e '' -a 1  -o 1.1 2.2 1.2  <(sort -k1,1 para.txt) <(sort -k1,1 para.title.txt) | sort -k1,1 -n > collection_with_title.tsv
#
#fi
#
#if [ -f "$ORIG_DIR/train.negatives.tsv" ]; then
#    echo "$ORIG_DIR/train.negatives.tsv exists.";
#else
#    echo "processing train.negatives.tsv -- negative documents for every query";
#    awk -v RS='\r\n' '$1==last {printf ",%s",$3; next} NR>1 {print "";} {last=$1; printf "%s\t%s",$1,$3;} END{print "";}' qidpidtriples.train.full.2.tsv > train.negatives.tsv
#
#fi
#echo "get raw data from original folder"
#mv $ORIG_DIR/train.negatives.tsv $RAW_DIR
#mv $ORIG_DIR/qrels.train.tsv $RAW_DIR/train.qrel.tsv
#mv $ORIG_DIR/train.query.txt $RAW_DIR
#mv $ORIG_DIR/collection_with_title.tsv $RAW_DIR/psg_corpus.tsv
#mv $ORIG_DIR/queries.dev.small.tsv $RAW_DIR/dev.query.txt
#mv $ORIG_DIR/qrels.dev.small.tsv $RAW_DIR/dev.qrel.trec


cd $CODE_DIR
export PYTHONPATH=.

if [ -f "$PROCESSED_DIR/train.jsonl" ]; then
    echo "$PROCESSED_DIR/train.jsonl exists";
else 
    echo "RUNNING build_train.py...";
    ### if you dont want the titles, change the template argument below to remove them
    python $CODE_DIR/scripts/msmarco/build_train.py \
        --tokenizer_name $PLM_DIR/t5-base-scaled  \
        --negative_file $RAW_DIR/train.negatives.tsv  \
        --qrels $RAW_DIR/train.qrel.tsv  \
        --queries $RAW_DIR/train.query.txt  \
        --collection $RAW_DIR/psg_corpus.tsv  \
        --save_to $PROCESSED_DIR/  \
        --template "Title: <title> Text: <text>" \
        --add_rand_negs

    echo "Concatenating output shards...";

    cd $PROCESSED_DIR/
    cat encoded_split-*.jsonl > train.jsonl
    rm encoded_split-*.jsonl

    tail -n 500 train.jsonl > val.jsonl
    head -n -500 train.jsonl > train_all.jsonl
    export RANDOM=42
    shuf -n 100000 train_all.jsonl > train.jsonl

    echo "Done setting up data and environments";

fi

### download and setup official trec eval scripts
echo "Setting up trec eval scripts";
cd $EVAL_DIR
if [ ! -d "$EVAL_DIR/trec_eval-9.0.7" ]; then
    wget https://trec.nist.gov/trec_eval/trec_eval-9.0.7.tar.gz
    tar -xvzf trec_eval-9.0.7.tar.gz
    rm tar -xvzf trec_eval-9.0.7.tar.gz
    cd trec_eval-9.0.7
    make;
    make quicktest;

fi

