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
#PROCESSED_DIR=$DATA_DIR"ance_hn_mt/mt_msmarco/processed/"$mt_method
if [ -d $MODEL_DIR ]; then
  echo "$MODEL_DIR is not empty"
else
  echo "get initial model"
  cp -r $TACO_DIR/model/ance_hn_mt/mt_msmarco/naive/ $MODEL_DIR
fi
#if [ -d $PROCESSED_DIR ]; then
#  echo "$PROCESSED_DIR is not empty"
#else
#  echo "get initial processed data"
#  mkdir -p $DATA_DIR"ance_hn_mt/mt_msmarco/processed/"
#  cp -r $DATA_DIR"ance_hn_mt/mt_msmarco/processed/naive/"  $PROCESSED_DIR
#fi

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

eval_delay=0
epoch=6
lr=1e-5
p_len=160
log_step=100
bsz=30
n_passages=8
infer_bsz=256
#mt_method="naive"
rands_ratio=0.5
n_gpu=8
num_hn_iters=6
epoch_per_hn=1
let last_hn_iter=${num_hn_iters}-1
echo "last hn iter ${last_hn_iter}"


for ((hn_iter=0; hn_iter<$num_hn_iters; hn_iter++))
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
    if [ ${mt_set} == zeshel ]; then
      max_q_len=128
    elif [ ${mt_set} == fever ]; then
      max_q_len=64
    else
      max_q_len=32
    fi
    if [ ${mt_set} == zeshel ]; then
      dev_corpus_path=$RAW_DIR/psg_corpus_dev.tsv
      train_corpus_path=$RAW_DIR/psg_corpus_train.tsv
      test_corpus_path=$RAW_DIR/psg_corpus_test.tsv
    elif [ ${mt_set} == nq ]; then
      dev_corpus_path=$DATA_DIR/kilt/corpus/psg_corpus.tsv
      train_corpus_path=$DATA_DIR/kilt/corpus/psg_corpus.tsv
      test_corpus_path=$DATA_DIR/kilt/corpus/psg_corpus.tsv
    else
      dev_corpus_path=$RAW_DIR/psg_corpus.tsv
      train_corpus_path=$RAW_DIR/psg_corpus.tsv
      test_corpus_path=$RAW_DIR/psg_corpus.tsv
    fi
    max_p_len=160
    n_passage=8
    if [ ${mt_set} == nq ]; then
      RAW_DIR=$DATA_DIR/kilt/${mt_set}/raw/
      NAIVE_INIT_DIR=$DATA_DIR/kilt/${mt_set}/processed/ance_hn_mt/mt_msmarco/naive/hn_iter_0
      PREFIX_PROCESSED=$DATA_DIR/kilt/${mt_set}/processed/ance_hn_mt/mt_msmarco/${mt_method}/
      PROCESSED_DIR=$PREFIX_PROCESSED/hn_iter_${hn_iter}/
    elif [ ${mt_set} == fever ]; then
      RAW_DIR=$DATA_DIR/beir/${mt_set}/raw/
      NAIVE_INIT_DIR=$DATA_DIR/beir/${mt_set}/processed/ance_hn_mt/mt_msmarco/naive/hn_iter_0
      PREFIX_PROCESSED=$DATA_DIR/beir/${mt_set}/processed/ance_hn_mt/mt_msmarco/${mt_method}/
      PROCESSED_DIR=$PREFIX_PROCESSED/hn_iter_${hn_iter}/
    else
      RAW_DIR=$DATA_DIR/${mt_set}/raw/
      PREFIX_PROCESSED=$DATA_DIR/${mt_set}/processed/ance_hn_mt/mt_msmarco/${mt_method}/
      NAIVE_INIT_DIR=$DATA_DIR/${mt_set}/processed/ance_hn_mt/mt_msmarco/naive/hn_iter_0
      PROCESSED_DIR=$PREFIX_PROCESSED/hn_iter_${hn_iter}/
    fi
    if [ $hn_iter == 0 ]; then
      echo "initial processed data should be obtained after warmup training"
      mkdir -p $PREFIX_PROCESSED
      if [ -d $PROCESSED_DIR ]; then
        echo "initial processed data already exists"
      else
        echo "copy from naive processed data"
        cp -r $NAIVE_INIT_DIR $PROCESSED_DIR
      fi
    fi
    train_path="$delimiter"$PROCESSED_DIR/train.jsonl
    val_path="$delimiter"$PROCESSED_DIR/val.jsonl
    mt_train_paths+=${train_path}
    mt_eval_paths+=${val_path}
    max_q_lens+="$delimiter"$max_q_len
    max_p_lens+="$delimiter"$max_p_len
    task_names+="$delimiter"${mt_set^^}
    mt_n_passages+="$delimiter"$n_passages

    echo "${mt_set} ance get train hard negatives for hn_iter ${hn_iter}"
    if [ $hn_iter != 0 ]; then
      if [ ${mt_set} == zeshel ]; then
        echo " build val hard negatives for zeshel"
        python src/taco/dataset/build_hn.py  \
            --tokenizer_name $PLM_DIR/t5-base-scaled  \
            --hn_file $RESULT_DIR/${mt_set}/hn_iter_${hn_iter}/dev.trec \
            --qrels $RAW_DIR/dev.qrel.tsv \
            --queries $RAW_DIR/dev.query.txt \
            --collection $dev_corpus_path \
            --save_to $PROCESSED_DIR \
            --template "Title: <title> Text: <text>" \
            --num_hards 32 \
            --num_rands 32 \
            --split dev \
            --seed 42 \
            --use_doc_id_map \
            --truncate $p_len
      fi
      echo "building train hard negatives of hn_iter ${hn_iter} for ${mt_set} ..."
      python src/taco/dataset/build_hn.py  \
          --tokenizer_name $PLM_DIR/t5-base-scaled  \
          --hn_file $RESULT_DIR/${mt_set}/hn_iter_${hn_iter}/train.trec \
          --qrels $RAW_DIR/train.qrel.tsv \
          --queries $train_query_path \
          --collection $train_corpus_path \
          --save_to $PROCESSED_DIR \
          --template "Title: <title> Text: <text>" \
          --num_hards 32 \
          --num_rands 32 \
          --split train \
          --seed ${hn_iter} \
          --use_doc_id_map \
          --truncate $p_len

      echo "removing training trec file of ${mt_set}"
      rm $RESULT_DIR/${mt_set}/hn_iter_${hn_iter}/train.trec
      echo "splitting ${mt_set} hn file"
      if [ ${mt_set} == zeshel ]; then
        mv $PROCESSED_DIR/train_all.jsonl  $PROCESSED_DIR/train.jsonl
        echo "splitting zeshel dev hn file"
        tail -n 500 $PROCESSED_DIR/dev_all.jsonl > $PROCESSED_DIR/val.jsonl
      else
        tail -n 500 $PROCESSED_DIR/train_all.jsonl > $PROCESSED_DIR/val.jsonl
        head -n -500 $PROCESSED_DIR/train_all.jsonl > $PROCESSED_DIR/train.jsonl
        rm $PROCESSED_DIR/train_all.jsonl
      fi
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
      --epochs_per_hn $epoch_per_hn \
      --logging_dir $LOG_DIR/hn_iter_${hn_iter}  \
      --negatives_x_device False \
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
  echo "building dev index for  episode-${hn_iter} "
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
    if [ ${mt_set} == zeshel ]; then
      max_q_len=128
    elif [ ${mt_set} == fever ]; then
      max_q_len=64
    else
      max_q_len=32
    fi
    if [ ${mt_set} == zeshel ]; then
      dev_corpus_path=$RAW_DIR/psg_corpus_dev.tsv
      train_corpus_path=$RAW_DIR/psg_corpus_train.tsv
      test_corpus_path=$RAW_DIR/psg_corpus_test.tsv
    elif [ ${mt_set} == nq ]; then
      dev_corpus_path=$DATA_DIR/kilt/corpus/psg_corpus.tsv
      train_corpus_path=$DATA_DIR/kilt/corpus/psg_corpus.tsv
      test_corpus_path=$DATA_DIR/kilt/corpus/psg_corpus.tsv
    else
      dev_corpus_path=$RAW_DIR/psg_corpus.tsv
      train_corpus_path=$RAW_DIR/psg_corpus.tsv
      test_corpus_path=$RAW_DIR/psg_corpus.tsv
    fi
    if [ ${mt_set} == nq ]; then
      RAW_DIR=$DATA_DIR/kilt/${mt_set}/raw/
      PROCESSED_DIR=$DATA_DIR/kilt/${mt_set}/processed/ance_hn_mt/mt_msmarco/${mt_method}/hn_iter_${new_hn_iter}/
    elif [ ${mt_set} == fever ]; then
      RAW_DIR=$DATA_DIR/beir/${mt_set}/raw/
      PROCESSED_DIR=$DATA_DIR/beir/${mt_set}/processed/ance_hn_mt/mt_msmarco/${mt_method}/hn_iter_${new_hn_iter}/
    else
      RAW_DIR=$DATA_DIR/${mt_set}/raw/
      PROCESSED_DIR=$DATA_DIR/${mt_set}/processed/ance_hn_mt/mt_msmarco/${mt_method}/hn_iter_${new_hn_iter}/
    fi
    mkdir -p $PROCESSED_DIR
    echo "build dev index for hn_iter ${hn_iter} ... "
    python src/taco/driver/build_index.py \
      --output_dir $EMBEDDING_DIR/ \
      --model_name_or_path $MODEL_DIR \
      --per_device_eval_batch_size $infer_bsz  \
      --corpus_path $dev_corpus_path  \
      --encoder_only False  \
      --doc_template "Title: <title> Text: <text>"  \
      --doc_column_names id,title,text \
      --q_max_len 32  \
      --p_max_len $p_len  \
      --fp16  \
      --dataloader_num_workers 0

    echo "retrieve dev data of ${mt_set} for hn_iter ${hn_iter} ... "
    if [ ! -d "$RESULT_DIR/${mt_set}/hn_iter_${new_hn_iter}/" ]; then
        mkdir -p $RESULT_DIR/${mt_set}/hn_iter_${new_hn_iter}/
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
        --trec_save_path $RESULT_DIR/${mt_set}/hn_iter_${new_hn_iter}/dev.trec \
        --dataloader_num_workers 0 \
        --task_name ${mt_set^^} \
        --add_query_task_prefix True

    $EVAL_DIR/trec_eval -c -mRprec -mrecip_rank.10 -mrecall.20,100 $RAW_DIR/dev.qrel.trec $RESULT_DIR/${mt_set}/hn_iter_${new_hn_iter}/dev.trec > $RESULT_DIR/${mt_set}/hn_iter_${new_hn_iter}/dev_results.txt
    if [ ${mt_set} == nq ]; then
      echo "page-level scoring ..."
      python scripts/kilt/convert_trec_to_provenance.py  \
        --trec_file $RESULT_DIR/${mt_set}/hn_iter_${new_hn_iter}/dev.trec  \
        --kilt_queries_file $RAW_DIR/${mt_set}-dev-kilt.jsonl  \
        --passage_collection $DATA_DIR/kilt/corpus/psgs_w100.tsv  \
        --output_provenance_file $RESULT_DIR/${mt_set}/hn_iter_${new_hn_iter}/provenance.json
      echo "get prediction file ... "
      python scripts/kilt/convert_to_evaluation.py \
        --kilt_queries_file $RAW_DIR/${mt_set}-dev-kilt.jsonl  \
        --provenance_file $RESULT_DIR/${mt_set}/hn_iter_${new_hn_iter}/provenance.json \
        --output_evaluation_file $RESULT_DIR/${mt_set}/hn_iter_${new_hn_iter}/preds.json
      echo "get scores ... "
      python scripts/kilt/evaluate_kilt.py $RESULT_DIR/${mt_set}/hn_iter_${new_hn_iter}/preds.json $RAW_DIR/${mt_set}-dev-kilt.jsonl \
        --ks 1,20,100 \
        --results_file $RESULT_DIR/${mt_set}/hn_iter_${new_hn_iter}/page-level-results.json
    fi
    if [ ${mt_set} == zeshel ] || [ ${mt_set} == fever ]; then
      echo "evaluate test data ... "
      if [ ${mt_set} == zeshel ]; then
        echo "build test index for zeshel"
        python src/taco/driver/build_index.py \
            --output_dir $EMBEDDING_DIR/ \
            --model_name_or_path $MODEL_DIR \
            --per_device_eval_batch_size $infer_bsz  \
            --corpus_path ${test_corpus_path}  \
            --encoder_only False  \
            --doc_template "Title: <title> Text: <text>"  \
            --doc_column_names id,title,text \
            --q_max_len $max_q_len  \
            --p_max_len $p_len  \
            --fp16  \
            --dataloader_num_workers 0
      fi
      echo "retrieve test ... "
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
          --trec_save_path $RESULT_DIR/${mt_set}/hn_iter_${new_hn_iter}/test.trec \
          --dataloader_num_workers 0 \
          --task_name ${mt_set^^} \
          --add_query_task_prefix True

      echo "evaluate test trec ... "
      $EVAL_DIR/trec_eval -c -mrecip_rank.10 -mrecall.64,100 $RAW_DIR/test.qrel.trec $RESULT_DIR/${mt_set}/hn_iter_${new_hn_iter}/test.trec > $RESULT_DIR/${mt_set}/hn_iter_${new_hn_iter}/test_results.txt

    fi

    if [ ${hn_iter} != ${last_hn_iter} ]; then
      if [ ${mt_set} == zeshel ]; then
        echo "build train index for zeshel ... "
        python src/taco/driver/build_index.py \
          --output_dir $EMBEDDING_DIR/ \
          --model_name_or_path $MODEL_DIR \
          --per_device_eval_batch_size $infer_bsz  \
          --corpus_path $train_corpus_path  \
          --encoder_only False  \
          --doc_template "Title: <title> Text: <text>"  \
          --doc_column_names id,title,text \
          --q_max_len $max_q_len  \
          --p_max_len $p_len  \
          --fp16  \
          --dataloader_num_workers 0
      fi
      echo "retrieve train trec"
      if [ ${mt_set} == msmarco ]; then
        echo "random down_sample msmarco"
        export RANDOM=${new_hn_iter}
        echo "random down_sample train queries ... "
        shuf -n 100000 $RAW_DIR/train.query.txt > $PROCESSED_DIR/train.query.txt
        train_query_path=$PROCESSED_DIR/train.query.txt
      else
        train_query_path=$RAW_DIR/train.query.txt
      fi
      echo "retrieving train ${mt_set} ..."
#      mkdir -p $RESULT_DIR/${mt_set}/hn_iter_${new_hn_iter}
      python -m src.taco.driver.retrieve  \
          --output_dir $EMBEDDING_DIR/ \
          --model_name_or_path $MODEL_DIR/ \
          --per_device_eval_batch_size $infer_bsz  \
          --query_path ${train_query_path}  \
          --encoder_only False  \
          --query_template "<text>"  \
          --query_column_names  id,text \
          --q_max_len $max_q_len  \
          --fp16  \
          --trec_save_path $RESULT_DIR/${mt_set}/hn_iter_${new_hn_iter}/train.trec \
          --dataloader_num_workers 0 \
          --topk 110 \
          --task_name ${mt_set^^} \
          --add_query_task_prefix True
    fi

  done
done
echo "deleting embedding cache"
rm $EMBEDDING_DIR/embeddings.corpus.rank.*
