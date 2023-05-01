#!/bin/bash
HOME_DIR="/common/users/wz283/projects/"
CODE_DIR=$HOME_DIR"/TACO/"
TACO_DIR=$HOME_DIR"/taco_data/"
PLM_DIR=$TACO_DIR"/plm/"
MODEL_DIR=$TACO_DIR"/model/beir/warmup_mt/"
DATA_DIR=$TACO_DIR"/data/beir/"
RAW_DIR=$DATA_DIR"/raw/"
PROCESSED_DIR=$DATA_DIR"/processed/bm25/"
LOG_DIR=$TACO_DIR"/logs/beir/warmup_mt/"
EMBEDDING_DIR=$TACO_DIR"/embeddings/warmup_mt/"
RESULT_DIR=$TACO_DIR"/results/warmup_mt/"
EVAL_DIR=$TACO_DIR"/metrics/trec/trec_eval-9.0.7/trec_eval-9.0.7/"
ANCE_PROCESSED_DIR=$DATA_DIR"/processed/ance_mt/"
ANCE_MODEL_DIR=$TACO_DIR"/model/beir/ance_mt/"
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
beir_sets=(trec-covid nfcorpus fiqa arguana webis-touche2020 quora scidocs scifact nq hotpotqa dbpedia-entity fever climate-fever)
has_train_sets=(nfcorpus hotpotqa fiqa fever scifact)
has_dev_sets=(nfcorpus hotpotqa fiqa quora dbpedia-entity fever)
has_train_dev_sets=(nfcorpus hotpotqa fiqa fever)

SAVE_STEP=10000
EVAL_STEP=300

eval_delay=30
epoch=40
lr=1e-5
p_len=160
log_step=100
bsz=64
n_passages=3
infer_bsz=256
mt_method="naive"
rands_ratio=0.5
n_gpu=8

mt_train_paths=""
mt_eval_paths=""
max_q_lens=""
max_p_lens=""
task_names=""
mt_n_passages=""
for beir_set in ${beir_sets[@]}
do
  if [ ${beir_set} == ${beir_sets[0]} ]; then
    delimiter=""
  else
    delimiter=","
  fi
  if [ ${beir_set} == wow ]; then
    max_q_len=256
  elif [ ${beir_set} == fever ]; then
    max_q_len=64
  elif [ ${beir_set} == aida ]; then
    max_q_len=128
  else
    max_q_len=32
  fi
  max_p_len=160
  n_passage=3
  mt_train_paths+="$delimiter"$PROCESSED_DIR/${beir_set}/train.jsonl
  mt_eval_paths+="$delimiter"$PROCESSED_DIR/${beir_set}/val.jsonl
  max_q_lens+="$delimiter"$max_q_len
  max_p_lens+="$delimiter"$max_p_len
  task_names+="$delimiter"${beir_set^^}
  mt_n_passages+="$delimiter"$n_passages
done

cd $CODE_DIR
export PYTHONPATH=.


echo "building index "
#  python src/taco/driver/build_index.py  \
torchrun --nproc_per_node=$n_gpu --standalone --nnodes=1 src/taco/driver/build_index.py \
    --output_dir $EMBEDDING_DIR/ \
    --model_name_or_path $MODEL_DIR \
    --per_device_eval_batch_size $infer_bsz  \
    --corpus_path $RAW_DIR/corpus/psg_corpus.tsv  \
    --encoder_only False  \
    --doc_template "Title: <title> Text: <text>"  \
    --doc_column_names id,title,text \
    --q_max_len 32  \
    --p_max_len $p_len  \
    --fp16  \
    --dataloader_num_workers 0

for beir_set in ${beir_sets[@]}
do
  if [ ${beir_set} == wow ]; then
    max_q_len=256
  elif [ ${beir_set} == fever ]; then
    max_q_len=64
  elif [ ${beir_set} == aida ]; then
    max_q_len=128
  else
    max_q_len=32
  fi
  echo "retrieve dev data of ${beir_set} ... "
  if [ ! -d "$RESULT_DIR/${beir_set}" ]; then
      mkdir -p $RESULT_DIR/${beir_set}
  fi

  python -m src.taco.driver.retrieve  \
      --output_dir $EMBEDDING_DIR/ \
      --model_name_or_path $MODEL_DIR \
      --per_device_eval_batch_size $infer_bsz  \
      --query_path $RAW_DIR/${beir_set}/dev.query.txt  \
      --encoder_only False  \
      --query_template "<text>"  \
      --query_column_names  id,text \
      --q_max_len $max_q_len  \
      --fp16  \
      --trec_save_path $RESULT_DIR/${beir_set}/dev.trec \
      --dataloader_num_workers 0

  $EVAL_DIR/trec_eval -c -mRprec -mrecip_rank.10 -mrecall.20,100 $RAW_DIR/${beir_set}/dev.qrel.trec $RESULT_DIR/${beir_set}/dev.trec > $RESULT_DIR/${beir_set}/dev_results.txt
  echo "page-level scoring ..."
  python scripts/beir/convert_trec_to_provenance.py  \
    --trec_file $RESULT_DIR/${beir_set}/dev.trec  \
    --beir_queries_file $RAW_DIR/${beir_set}/${beir_set}-dev-beir.jsonl  \
    --passage_collection $RAW_DIR/corpus/psgs_w100.tsv  \
    --output_provenance_file $RESULT_DIR/${beir_set}/provenance.json
  echo "get prediction file ... "
  python scripts/beir/convert_to_evaluation.py \
    --beir_queries_file $RAW_DIR/${beir_set}/${beir_set}-dev-beir.jsonl  \
    --provenance_file $RESULT_DIR/${beir_set}/provenance.json \
    --output_evaluation_file $RESULT_DIR/${beir_set}/preds.json
  echo "get scores ... "
  python scripts/beir/evaluate_beir.py $RESULT_DIR/${beir_set}/preds.json $RAW_DIR/${beir_set}/${beir_set}-dev-beir.jsonl \
    --ks 1,20,100 \
    --results_file $RESULT_DIR/${beir_set}/page-level-results.json

  echo "get preprocessed data of ${beir_set} for ance training"
  echo "retrieving train ..."
  python -m src.taco.driver.retrieve  \
      --output_dir $EMBEDDING_DIR/ \
      --model_name_or_path $MODEL_DIR \
      --per_device_eval_batch_size $infer_bsz  \
      --query_path $RAW_DIR/${beir_set}/train.query.txt  \
      --encoder_only False  \
      --query_template "<text>"  \
      --query_column_names  id,text \
      --q_max_len $max_q_len  \
      --fp16  \
      --trec_save_path $RESULT_DIR/${beir_set}/train.trec \
      --dataloader_num_workers 0 \
      --topk 110

  echo "building hard negatives of ance first episode for ${beir_set} ..."
  mkdir -p $ANCE_PROCESSED_DIR/${beir_set}/hn_iter_0
  python src/taco/dataset/build_hn.py  \
      --tokenizer_name $PLM_DIR/t5-base-scaled  \
      --hn_file $RESULT_DIR/${beir_set}/train.trec \
      --qrels $RAW_DIR/${beir_set}/train.qrel.tsv \
      --queries $RAW_DIR/${beir_set}/train.query.txt \
      --collection $RAW_DIR/corpus/psg_corpus.tsv \
      --save_to $ANCE_PROCESSED_DIR/${beir_set}/hn_iter_0 \
      --template "Title: <title> Text: <text>" \
      --add_rand_negs True \
      --num_hards 64 \
      --num_rands 64 \
      --use_doc_id_map True

  echo "removing train ${beir_set} trec files"
  rm $RESULT_DIR/${beir_set}/train.trec

  echo "splitting ${beir_set} hn file"
  tail -n 500 $ANCE_PROCESSED_DIR/${beir_set}/hn_iter_0/train_all.jsonl > $ANCE_PROCESSED_DIR/${beir_set}/hn_iter_0/val.jsonl
  head -n -500 $ANCE_PROCESSED_DIR/${beir_set}/hn_iter_0/train_all.jsonl > $ANCE_PROCESSED_DIR/${beir_set}/hn_iter_0/train.jsonl
  rm $ANCE_PROCESSED_DIR/${beir_set}/hn_iter_0/train_all.jsonl
done

echo "moving warmed up model to ance iter 0 model folder"
mv $MODEL_DIR  $ANCE_MODEL_DIR/hn_iter_0/
echo "deleting warmed up embeddings ... "
rm $EMBEDDING_DIR/embeddings.corpus.rank.*
