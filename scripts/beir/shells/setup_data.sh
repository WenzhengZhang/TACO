#!/bin/bash

HOME_DIR="/common/users/wz283/projects/"
CODE_DIR=$HOME_DIR"/TACO/"
TACO_DIR=$HOME_DIR"/taco_data/"
PLM_DIR=$TACO_DIR"/plm/"
MODEL_DIR=$TACO_DIR"/model/beir"
DATA_DIR=$TACO_DIR"/data/beir/"
ORIG_DIR=$DATA_DIR"/original/"
#RAW_DIR=$DATA_DIR"/raw/"
#PROCESSED_DIR=$DATA_DIR"/processed/"
mkdir -p $TACO_DIR
mkdir -p $PLM_DIR
mkdir -p $MODEL_DIR
mkdir -p $DATA_DIR
#mkdir -p $RAW_DIR
#mkdir -p $PROCESSED_DIR
mkdir -p $EVAL_DIR
mkdir -p $ORIG_DIR
cd $ORIG_DIR
#beir_sets=(trec-covid nfcorpus fiqa arguana webis-touche2020 quora scidocs scifact nq hotpotqa dbpedia-entity fever climate-fever)
beir_sets=(nq-train)
#has_train_sets=(nfcorpus hotpotqa fiqa fever scifact)
has_train_sets=(nq-train)
#has_dev_sets=(nfcorpus hotpotqa fiqa quora dbpedia-entity fever)
has_dev_sets=(nq-train)
#has_train_dev_sets=(nfcorpus hotpotqa fiqa fever)
has_train_dev_sets=(nq-train)
for dataset in ${beir_sets[@]}
do
  if [ -d "$DATA_DIR/${dataset}/raw/" ]; then
    echo "$DATA_DIR/${dataset}/raw already exists.";
  else
    echo "downloading ${dataset}"
    mkdir -p $DATA_DIR/${dataset}/raw/
  #  mkdir -p $PROCESSED_DIR/"bm25/"$dataset
    python $CODE_DIR/scripts/beir/download_data.py --dataset_name ${dataset} \
      --out_dir $ORIG_DIR
    echo "process original ${dataset} to TACO format"
    if [[ " ${has_train_dev_sets[*]} " =~ " ${dataset} " ]]; then
      echo "process both train,dev and test"
      python $CODE_DIR/scripts/beir/preprocess_data.py --input_dir $ORIG_DIR \
        --processed_dir $DATA_DIR/${dataset}/raw \
        --process_train \
        --process_dev \
        --dataset_name ${dataset}
    elif [[ " ${has_train_sets[*]} " =~ " ${dataset} " ]]; then
      echo "process train and test"
      python $CODE_DIR/scripts/beir/preprocess_data.py --input_dir $ORIG_DIR \
        --processed_dir $DATA_DIR/${dataset}/raw \
        --process_train \
        --dataset_name ${dataset}
    elif [[ " ${has_dev_sets[*]} " =~ " ${dataset} " ]]; then
      echo "process dev and test"
      python $CODE_DIR/scripts/beir/preprocess_data.py --input_dir $ORIG_DIR \
        --processed_dir $DATA_DIR/${dataset}/raw \
        --process_dev \
        --dataset_name ${dataset}
    else
      echo "process test"
      python $CODE_DIR/scripts/beir/preprocess_data.py --input_dir $ORIG_DIR \
        --processed_dir $DATA_DIR/${dataset}/raw \
        --dataset_name ${dataset}
    fi
    echo "remove original beir data"
    rm -rf $ORIG_DIR/${dataset}
  fi
done
