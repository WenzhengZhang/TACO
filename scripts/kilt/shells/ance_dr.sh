#!/bin/bash
HOME_DIR="/common/users/wz283/projects/"
CODE_DIR=$HOME_DIR"/TACO/"
TACO_DIR=$HOME_DIR"/taco_data/"
PLM_DIR=$TACO_DIR"/plm/"
MODEL_DIR=$TACO_DIR"/model/kilt/ance_dr/"
DATA_DIR=$TACO_DIR"/data/kilt/"
#RAW_DIR=$DATA_DIR"/raw/"
#PROCESSED_DIR=$DATA_DIR"/processed/ance_dr/"
LOG_DIR=$TACO_DIR"/logs/kilt/ance_dr/"
EMBEDDING_DIR=$TACO_DIR"/embeddings/ance_dr/"
RESULT_DIR=$TACO_DIR"/results/ance_dr/"
EVAL_DIR=$TACO_DIR"/metrics/trec/trec_eval-9.0.7/trec_eval-9.0.7/"
mkdir -p $TACO_DIR
mkdir -p $PLM_DIR
mkdir -p $MODEL_DIR
mkdir -p $DATA_DIR
#mkdir -p $RAW_DIR
#mkdir -p $PROCESSED_DIR
mkdir -p $LOG_DIR
mkdir -p $EMBEDDING_DIR
mkdir -p $RESULT_DIR
mkdir -p $EVAL_DIR
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
kilt_sets=(nq tqa hopo wow trex fever zsre aida)

SAVE_STEP=10000
EVAL_STEP=300

eval_delay=0
epoch=3
p_len=160
log_step=100
n_passages=3

num_hn_iters=8
epoch_per_hn=3
lr=5e-6
dr=1
n_gpu=8
bsz=32
infer_bsz=256
steps=250
n_gpu=8

for kilt_set in ${kilt_sets[@]}
do
  if [ ${kilt_set} == wow ]; then
    max_q_len=256
  elif [ ${kilt_set} == fever ]; then
    max_q_len=64
  elif [ ${kilt_set} == aida ]; then
    max_q_len=128
  else
    max_q_len=32
  fi
  echo "${kilt_set} ance training"
  for ((hn_iter=0; hn_iter<$num_hn_iters; hn_iters++))
  do
      echo "Iteration $hn_iter"
      let new_hn_iter=$hn_iter+1
      if [ $hn_iter != 0 ]; then
        echo "retrieving train ..."
        mkdir -p $RESULT_DIR/${kilt_set}/hn_iter_${hn_iter}
        python -m src.taco.driver.retrieve  \
            --output_dir $EMBEDDING_DIR/ \
            --model_name_or_path $MODEL_DIR/${kilt_set}/hn_iter_${hn_iter} \
            --per_device_eval_batch_size $infer_bsz  \
            --query_path $DATA_DIR/${kilt_set}/raw/train.query.txt  \
            --encoder_only False  \
            --query_template "<text>"  \
            --query_column_names  id,text \
            --q_max_len $max_q_len  \
            --fp16  \
            --trec_save_path $RESULT_DIR/${kilt_set}/hn_iter_${hn_iter}/train.trec \
            --dataloader_num_workers 0 \
            --topk 110
        echo "building hard negatives of ance first episode for ${kilt_set} ..."
        mkdir -p $DATA_DIR/${kilt_set}/processed/ance_dr/hn_iter_${hn_iter}
        python src/taco/dataset/build_hn.py  \
            --tokenizer_name $PLM_DIR/t5-base-scaled  \
            --hn_file $RESULT_DIR/${kilt_set}/hn_iter_${hn_iter}/train.trec \
            --qrels $DATA_DIR/${kilt_set}/raw/train.qrel.tsv \
            --queries $DATA_DIR/${kilt_set}/raw/train.query.txt \
            --collection $DATA_DIR/corpus/psg_corpus.tsv \
            --save_to $DATA_DIR/${kilt_set}/processed/ance_dr/hn_iter_${hn_iter} \
            --template "Title: <title> Text: <text>" \
            --add_rand_negs True \
            --num_hards 64 \
            --num_rands 64
  
        echo "removing training trec file of ${kilt_set}"
        rm $RESULT_DIR/${kilt_set}/hn_iter_${hn_iter}/train.trec
        echo "splitting ${kilt_set} hn file"
        tail -n 500 $DATA_DIR/${kilt_set}/processed/ance_dr/hn_iter_${hn_iter}/train_all.jsonl > $DATA_DIR/${kilt_set}/processed/ance_dr/hn_iter_${hn_iter}/val.jsonl
        head -n -500 $DATA_DIR/${kilt_set}/processed/ance_dr/hn_iter_${hn_iter}/train_all.jsonl > $DATA_DIR/${kilt_set}/processed/ance_dr/hn_iter_${hn_iter}/train.jsonl
        rm $DATA_DIR/${kilt_set}/processed/ance_dr/hn_iter_${hn_iter}/train_all.jsonl
      fi


      echo "start hn training for ${kilt_set} for episode-${hn_iter} ..."

      torchrun --nproc_per_node=$n_gpu --standalone --nnodes=1 src/taco/driver/train_dr.py \
          --output_dir $MODEL_DIR/${kilt_set}/hn_iter_${new_hn_iter}  \
          --model_name_or_path $MODEL_DIR/${kilt_set}/hn_iter_${hn_iter}  \
          --do_train  \
          --eval_delay $eval_delay \
          --save_strategy epoch \
          --evaluation_strategy epoch \
          --logging_steps $log_step \
          --train_path $DATA_DIR/${kilt_set}/processed/ance_dr/hn_iter_${hn_iter}/train.jsonl  \
          --eval_path $DATA_DIR/${kilt_set}/processed/ance_dr/hn_iter_${hn_iter}/val.jsonl \
          --fp16  \
          --per_device_train_batch_size $bsz  \
          --train_n_passages $n_passages  \
          --learning_rate $lr  \
          --q_max_len $max_q_len  \
          --p_max_len $p_len \
          --num_train_epochs $epoch  \
          --logging_dir $LOG_DIR/${kilt_set}/hn_iter_${hn_iter}  \
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
          --load_best_model_at_end False \
          --metric_for_best_model loss

      echo "evaluating ${kilt_set} for episode-${hn_iter} ..."
      echo "building index for ${kilt_set} for episode-${hn_iter} "
      #  python src/taco/driver/build_index.py  \
      torchrun --nproc_per_node=$n_gpu --standalone --nnodes=1 src/taco/driver/build_index.py \
          --output_dir $EMBEDDING_DIR/ \
          --model_name_or_path $MODEL_DIR/${kilt_set}/hn_iter_${new_hn_iter} \
          --per_device_eval_batch_size $infer_bsz  \
          --corpus_path $DATA_DIR/corpus/psg_corpus.tsv  \
          --encoder_only False  \
          --doc_template "Title: <title> Text: <text>"  \
          --doc_column_names id,title,text \
          --q_max_len $max_q_len  \
          --p_max_len $p_len  \
          --fp16  \
          --dataloader_num_workers 0

      echo "retrieve dev data of ${kilt_set} for episode-${hn_iter} ... "
      if [ ! -d "$RESULT_DIR/${kilt_set}/hn_iter_${hn_iter}" ]; then
          mkdir -p $RESULT_DIR/${kilt_set}/hn_iter_${hn_iter}
      fi

      python -m src.taco.driver.retrieve  \
          --output_dir $EMBEDDING_DIR/ \
          --model_name_or_path $MODEL_DIR/${kilt_set}/hn_iter_${new_hn_iter} \
          --per_device_eval_batch_size $infer_bsz  \
          --query_path $DATA_DIR/${kilt_set}/raw/dev.query.txt  \
          --encoder_only False  \
          --query_template "<text>"  \
          --query_column_names  id,text \
          --q_max_len $max_q_len  \
          --fp16  \
          --trec_save_path $RESULT_DIR/${kilt_set}/hn_iter_${hn_iter}/dev.trec \
          --dataloader_num_workers 0

      $EVAL_DIR/trec_eval -c -mRprec -mrecip_rank.10 -mrecall.20,100 $DATA_DIR/${kilt_set}/raw/dev.qrel.trec $RESULT_DIR/${kilt_set}/hn_iter_${hn_iter}/dev.trec > $RESULT_DIR/${kilt_set}/hn_iter_${hn_iter}/dev_results.txt
      echo "page-level scoring ..."
      python scripts/kilt/convert_trec_to_provenance.py  \
        --trec_file $RESULT_DIR/${kilt_set}/hn_iter_${hn_iter}/dev.trec  \
        --kilt_queries_file $DATA_DIR/${kilt_set}/raw/${kilt_set}-dev-kilt.jsonl  \
        --passage_collection $DATA_DIR/corpus/psgs_w100.tsv  \
        --output_provenance_file $RESULT_DIR/${kilt_set}/hn_iter_${hn_iter}/provenance.json
      echo "get prediction file ... "
      python scripts/kilt/convert_to_evaluation.py \
        --kilt_queries_file $DATA_DIR/${kilt_set}/raw/${kilt_set}-dev-kilt.jsonl  \
        --provenance_file $RESULT_DIR/${kilt_set}/hn_iter_${hn_iter}/provenance.json \
        --output_evaluation_file $RESULT_DIR/${kilt_set}/hn_iter_${hn_iter}/preds.json
      echo "get scores ... "
      python scripts/kilt/evaluate_kilt.py $RESULT_DIR/${kilt_set}/hn_iter_${hn_iter}/preds.json $DATA_DIR/${kilt_set}/raw/${kilt_set}-dev-kilt.jsonl \
        --ks 1,20,100 \
        --results_file $RESULT_DIR/${kilt_set}/hn_iter_${hn_iter}/page-level-results.json
  done
done
echo "deleting embedding cache"
rm $EMBEDDING_DIR/embeddings.corpus.rank.*
