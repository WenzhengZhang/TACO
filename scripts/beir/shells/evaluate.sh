#!/bin/bash
MODEL_DIR=$1
MODEL_TYPE=$2
HOME_DIR="/common/users/wz283/projects/"
CODE_DIR=$HOME_DIR"/TACO/"
TACO_DIR=$HOME_DIR"/taco_data/"
PLM_DIR=$TACO_DIR"/plm/"
DATA_DIR=$TACO_DIR"/data/beir/"
RAW_DIR=$DATA_DIR"/raw/"
PROCESSED_DIR=$DATA_DIR"/processed/"
LOG_DIR=$TACO_DIR"/logs/beir/"
EMBEDDING_DIR=$TACO_DIR"/embeddings/"
RESULT_DIR=$TACO_DIR"/results/beir/$MODEL_TYPE/"
EVAL_DIR=$TACO_DIR"/metrics/trec/trec_eval-9.0.7/trec_eval-9.0.7/"
#ANCE_PROCESSED_DIR=$DATA_DIR"/processed/ance_dr/"
#ANCE_MODEL_DIR=$TACO_DIR"/model/beir/ance_dr/"
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
#mkdir -p $ANCE_MODEL_DIR
#mkdir -p $ANCE_PROCESSED_DIR
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
beir_sets=(trec-covid nfcorpus fiqa arguana webis-touche2020 quora scidocs scifact nq hotpotqa dbpedia-entity fever climate-fever)

SAVE_STEP=10000
EVAL_STEP=300

eval_delay=30
epoch=40
lr=1e-5
p_len=160
log_step=100
bsz=16
n_passages=3
infer_bsz=1024
n_gpu=8

cd $CODE_DIR
export PYTHONPATH=.

for beir_set in ${beir_sets[@]}
do
  echo "setup max query and doc length"
  if [ ${beir_set} == arguana ]; then
    max_q_len=132
  else
    max_q_len=68
  fi
  if [ ${beir_set} == scifact ] || [ ${beir_set} == trec-news ]; then
    p_len=260
  else
    p_len=160
  fi
  echo "building index for ${beir_set}"
#  python src/taco/driver/build_index.py  \
  torchrun --nproc_per_node=$n_gpu --standalone --nnodes=1 src/taco/driver/build_index.py \
      --output_dir $EMBEDDING_DIR/ \
      --model_name_or_path $MODEL_DIR \
      --per_device_eval_batch_size $infer_bsz  \
      --corpus_path $RAW_DIR/${beir_set}/corpus.tsv  \
      --encoder_only False  \
      --doc_template "Title: <title> Text: <text>"  \
      --doc_column_names id,title,text \
      --q_max_len $max_q_len  \
      --p_max_len $p_len  \
      --fp16  \
      --dataloader_num_workers 0

  echo "retrieve test data of ${beir_set} ... "
  if [ ! -d "$RESULT_DIR/${beir_set}" ]; then
      mkdir -p $RESULT_DIR/${beir_set}
  fi

  python -m src.taco.driver.retrieve  \
      --output_dir $EMBEDDING_DIR/ \
      --model_name_or_path $MODEL_DIR \
      --per_device_eval_batch_size $infer_bsz  \
      --query_path $RAW_DIR/${beir_set}/queries.test.tsv  \
      --encoder_only False  \
      --query_template "<text>"  \
      --query_column_names  id,text \
      --q_max_len $max_q_len  \
      --fp16  \
      --trec_save_path $RESULT_DIR/${beir_set}/test.trec \
      --dataloader_num_workers 0 \
      --task_name ${beir_set^^} \
      --add_query_task_prefix True

  $EVAL_DIR/trec_eval -c -mrecip_rank.10 -mndcg_cut.10 $RAW_DIR/${beir_set}/qrel.test.trec $RESULT_DIR/${beir_set}/test.trec > $RESULT_DIR/${beir_set}/test_results.txt
 
  echo "deleting warmed up embeddings for ${beir_set}"
  rm $EMBEDDING_DIR/embeddings.corpus.rank.*

done

