#!/bin/bash
mt_method=$1
HOME_DIR="/common/users/wz283/projects/"
CACHE_DIR="/common/users/wz283/hf_dataset_cache/"
CODE_DIR=$HOME_DIR"/TACO/"
TACO_DIR=$HOME_DIR"/taco_data/"
PLM_DIR=$TACO_DIR"/plm/"
MODEL_DIR=$TACO_DIR"/model/ance_hn_mt/mt_msmarco/"$mt_method
DATA_DIR=$TACO_DIR"/data/"
#RAW_DIR=$DATA_DIR"/raw/"
#PROCESSED_DIR=$DATA_DIR"/processed/bm25/"
LOG_DIR=$TACO_DIR"/logs/ance_hn_mt/mt_msmarco/"$mt_method
EMBEDDING_DIR=$TACO_DIR"/embeddings/ance_hn_mt/mt_msmarco/"$mt_method
RESULT_DIR=$TACO_DIR"/results/ance_hn_mt/mt_msmarco/"$mt_method
EVAL_DIR=$TACO_DIR"/metrics/trec/trec_eval-9.0.7/trec_eval-9.0.7/"
PROCESSED_DIR=$DATA_DIR"ance_hn_mt/mt_msmarco/processed/"$mt_method
if [ -d $MODEL_DIR ]; then
  echo "$MODEL_DIR is not empty"
else
  echo "get initial model"
  cp -r $TACO_DIR/model/ance_hn_mt/mt_msmarco/naive/ $MODEL_DIR
fi
if [ -d $PROCESSED_DIR ]; then
  echo "$PROCESSED_DIR is not empty"
else
  echo "get initial processed data"
  mkdir -p $DATA_DIR"ance_hn_mt/mt_msmarco/processed/"
  cp -r $DATA_DIR"ance_hn_mt/mt_msmarco/processed/naive/"  $PROCESSED_DIR
fi

mkdir -p $TACO_DIR
mkdir -p $PLM_DIR
#mkdir -p $MODEL_DIR
mkdir -p $DATA_DIR
#mkdir -p $RAW_DIR
#mkdir -p $PROCESSED_DIR
mkdir -p $LOG_DIR
mkdir -p $EMBEDDING_DIR
mkdir -p $RESULT_DIR
mkdir -p $EVAL_DIR
#mkdir -p $ANCE_MODEL_DIR
#mkdir -p $ANCE_PROCESSED_DIR
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mt_sets=(msmarco nq zeshel fever)

SAVE_STEP=10000
EVAL_STEP=300

eval_delay=30
epoch=40
lr=1e-5
p_len=160
log_step=100
bsz=32
n_passages=8
infer_bsz=256
#mt_method="naive"
rands_ratio=0.5
n_gpu=8



for ((hn_iter=0; hn_iter<$num_hn_iters; hn_iters++))
do
  echo "ance episode $hn_iter"
  if [ $hn_iter != 0 ]; then
    resume=$(ls -td $MODEL_DIR/checkpoint-* | head -1)
  else
    resume=False
  fi
  let new_hn_iter=$hn_iter+1
  mt_train_paths=""
  mt_eval_paths=""
  max_q_lens=""
  max_p_lens=""
  task_names=""
  mt_n_passages=""
  for mt_set in ${mt_sets[@]}
  do
    if [ ${mt_set} == ${mt_sets[0]} ]; then
      delimiter=""
    else
      delimiter=","
    fi
    if [ ${mt_set} == wow ]; then
      max_q_len=260
    elif [ ${mt_set} == fever ]; then
      max_q_len=68
    elif [ ${mt_set} == aida ]; then
      max_q_len=132
    else
      max_q_len=36
    fi
    max_p_len=160
    n_passage=3
    mt_train_paths+="$delimiter"$DATA_DIR/${mt_set}/processed/ance_hn_mt/${mt_method}/hn_iter_${hn_iter}/train.jsonl
    mt_eval_paths+="$delimiter"$DATA_DIR/${mt_set}/processed/ance_hn_mt/${mt_method}/hn_iter_${hn_iter}/val.jsonl
    max_q_lens+="$delimiter"$max_q_len
    max_p_lens+="$delimiter"$max_p_len
    task_names+="$delimiter"${mt_set^^}
    mt_n_passages+="$delimiter"$n_passages
  done
  for mt_set in ${mt_sets[@]}
  do
    echo "${mt_set} ance training"
    if [ $hn_iter != 0 ]; then
      echo "retrieving train ${mt_set} ..."
      mkdir -p $RESULT_DIR/${mt_set}/hn_iter_${hn_iter}
      python -m src.taco.driver.retrieve  \
          --output_dir $EMBEDDING_DIR/ \
          --model_name_or_path $MODEL_DIR/ \
          --per_device_eval_batch_size $infer_bsz  \
          --query_path $DATA_DIR/${mt_set}/raw/train.query.txt  \
          --encoder_only False  \
          --query_template "<text>"  \
          --query_column_names  id,text \
          --q_max_len $max_q_len  \
          --fp16  \
          --trec_save_path $RESULT_DIR/${mt_set}/hn_iter_${hn_iter}/train.trec \
          --dataloader_num_workers 0 \
          --topk 110 \
          --task_name ${mt_set^^} \
          --add_query_task_prefix True
      echo "building hard negatives of ance first episode for ${mt_set} ..."
      mkdir -p $DATA_DIR/${mt_set}/processed/ance_hn_mt/${mt_method}/hn_iter_${hn_iter}
      python src/taco/dataset/build_hn.py  \
          --tokenizer_name $PLM_DIR/t5-base-scaled  \
          --hn_file $RESULT_DIR/${mt_set}/hn_iter_${hn_iter}/train.trec \
          --qrels $DATA_DIR/${mt_set}/raw/train.qrel.tsv \
          --queries $DATA_DIR/${mt_set}/raw/train.query.txt \
          --collection $DATA_DIR/corpus/psg_corpus.tsv \
          --save_to $DATA_DIR/${mt_set}/processed/ance_hn_mt/${mt_method}/hn_iter_${hn_iter} \
          --template "Title: <title> Text: <text>" \
          --add_rand_negs True \
          --num_hards 64 \
          --num_rands 64

      echo "removing training trec file of ${mt_set}"
      rm $RESULT_DIR/${mt_set}/hn_iter_${hn_iter}/train.trec
      echo "splitting ${mt_set} hn file"
      tail -n 500 $DATA_DIR/${mt_set}/processed/ance_hn_mt/${mt_method}/hn_iter_${hn_iter}/train_all.jsonl > $DATA_DIR/${mt_set}/processed/ance_hn_mt/${mt_method}/hn_iter_${hn_iter}/val.jsonl
      head -n -500 $DATA_DIR/${mt_set}/processed/ance_hn_mt/${mt_method}/hn_iter_${hn_iter}/train_all.jsonl > $DATA_DIR/${mt_set}/processed/ance_hn_mt/${mt_method}/hn_iter_${hn_iter}/train.jsonl
      rm $DATA_DIR/${mt_set}/processed/ance_hn_mt/${mt_method}/hn_iter_${hn_iter}/train_all.jsonl
    fi
  done


  echo "start hn training for for episode-${hn_iter} ..."

  torchrun --nproc_per_node=$n_gpu --standalone --nnodes=1 src/taco/driver/train_mt.py \
      --output_dir $MODEL_DIR  \
      --model_name_or_path $MODEL_DIR  \
      --do_train  \
      --eval_delay $eval_delay \
      --save_strategy epoch \
      --evaluation_strategy epoch \
      --logging_steps $log_step \
      --mt_train_paths $mt_train_paths  \
      --mt_eval_paths $mt_eval_paths \
      --fp16  \
      --per_device_train_batch_size $bsz  \
      --mt_train_n_passages $mt_n_passages \
      --learning_rate $lr  \
      --q_max_lens $max_q_lens  \
      --p_max_lens $max_p_lens \
      --task_names $task_names \
      --num_train_epochs $epoch  \
      --logging_dir $LOG_DIR/hn_iter_${hn_iter}  \
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
      --metric_for_best_model loss \
      --up_sample True \
      --weight_method $mt_method \
      --select_all True \
      --multi_mix_temp 4.0 \
      --add_query_task_prefix True \
      --log_gnorm False \
      --beta_taco 0.999 \
      --tau_taco 2 \
      --beta_gn 1.5 \
      --beta_cgd 0.25 \
      --tau_cgd 100 \
      --norm_grad True \
      --norm_ipt True \
      --hard_negative_mining True \
      --rands_ratio $rands_ratio \
      --add_query_task_prefix True \
      --resume_from_checkpoint $resume

  echo "evaluating for episode-${hn_iter} ..."
  echo "building index for  episode-${hn_iter} "
  #  python src/taco/driver/build_index.py  \
  torchrun --nproc_per_node=$n_gpu --standalone --nnodes=1 src/taco/driver/build_index.py \
      --output_dir $EMBEDDING_DIR/ \
      --model_name_or_path $MODEL_DIR \
      --per_device_eval_batch_size $infer_bsz  \
      --corpus_path $DATA_DIR/corpus/psg_corpus.tsv  \
      --encoder_only False  \
      --doc_template "Title: <title> Text: <text>"  \
      --doc_column_names id,title,text \
      --q_max_len 32  \
      --p_max_len $p_len  \
      --fp16  \
      --dataloader_num_workers 0

  for mt_set in ${mt_sets[@]}
  do
    if [ ${mt_set} == wow ]; then
    max_q_len=256
    elif [ ${mt_set} == fever ]; then
      max_q_len=64
    elif [ ${mt_set} == aida ]; then
      max_q_len=128
    else
      max_q_len=32
    fi
    echo "retrieve dev data of ${mt_set} ... "
    if [ ! -d "$RESULT_DIR/${mt_set}" ]; then
        mkdir -p $RESULT_DIR/${mt_set}
    fi
    echo "retrieve dev data of ${mt_set} for episode-${hn_iter} ... "
    if [ ! -d "$RESULT_DIR/${mt_set}/hn_iter_${hn_iter}" ]; then
        mkdir -p $RESULT_DIR/${mt_set}/hn_iter_${hn_iter}
    fi

    python -m src.taco.driver.retrieve  \
        --output_dir $EMBEDDING_DIR/ \
        --model_name_or_path $MODEL_DIR \
        --per_device_eval_batch_size $infer_bsz  \
        --query_path $DATA_DIR/${mt_set}/raw/dev.query.txt  \
        --encoder_only False  \
        --query_template "<text>"  \
        --query_column_names  id,text \
        --q_max_len $max_q_len  \
        --fp16  \
        --trec_save_path $RESULT_DIR/${mt_set}/hn_iter_${hn_iter}/dev.trec \
        --dataloader_num_workers 0 \
        --task_name ${mt_set^^} \
        --add_query_task_prefix True

    $EVAL_DIR/trec_eval -c -mRprec -mrecip_rank.10 -mrecall.20,100 $DATA_DIR/${mt_set}/raw/dev.qrel.trec $RESULT_DIR/${mt_set}/hn_iter_${hn_iter}/dev.trec > $RESULT_DIR/${mt_set}/hn_iter_${hn_iter}/dev_results.txt
    echo "page-level scoring ..."
    python scripts/kilt/convert_trec_to_provenance.py  \
      --trec_file $RESULT_DIR/${mt_set}/hn_iter_${hn_iter}/dev.trec  \
      --kilt_queries_file $DATA_DIR/${mt_set}/raw/${mt_set}-dev-kilt.jsonl  \
      --passage_collection $DATA_DIR/corpus/psgs_w100.tsv  \
      --output_provenance_file $RESULT_DIR/${mt_set}/hn_iter_${hn_iter}/provenance.json
    echo "get prediction file ... "
    python scripts/kilt/convert_to_evaluation.py \
      --kilt_queries_file $DATA_DIR/${mt_set}/raw/${mt_set}-dev-kilt.jsonl  \
      --provenance_file $RESULT_DIR/${mt_set}/hn_iter_${hn_iter}/provenance.json \
      --output_evaluation_file $RESULT_DIR/${mt_set}/hn_iter_${hn_iter}/preds.json
    echo "get scores ... "
    python scripts/kilt/evaluate_kilt.py $RESULT_DIR/${mt_set}/hn_iter_${hn_iter}/preds.json $DATA_DIR/${mt_set}/raw/${mt_set}-dev-kilt.jsonl \
      --ks 1,20,100 \
      --results_file $RESULT_DIR/${mt_set}/hn_iter_${hn_iter}/page-level-results.json
  done
done
echo "deleting embedding cache"
rm $EMBEDDING_DIR/embeddings.corpus.rank.*
