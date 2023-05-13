#!/bin/bash
HOME_DIR="/common/users/wz283/projects/"
CACHE_DIR="/common/users/wz283/hf_dataset_cache/"
CODE_DIR=$HOME_DIR"/TACO/"
TACO_DIR=$HOME_DIR"/taco_data/"
PLM_DIR=$TACO_DIR"/plm/"
MODEL_DIR=$TACO_DIR"/model/warmup_mt/mt_msmarco/"
DATA_DIR=$TACO_DIR"/data/"
#RAW_DIR=$DATA_DIR"/raw/"
#PROCESSED_DIR=$DATA_DIR"/processed/bm25/"
LOG_DIR=$TACO_DIR"/logs/warmup_mt/mt_msmarco/"
EMBEDDING_DIR=$TACO_DIR"/embeddings/warmup_mt/mt_msmarco/"
RESULT_DIR=$TACO_DIR"/results/warmup_mt/mt_msmarco/"
EVAL_DIR=$TACO_DIR"/metrics/trec/trec_eval-9.0.7/trec_eval-9.0.7/"
#ANCE_PROCESSED_DIR=$DATA_DIR"ance_hn_mt/mt_msmarco/processed/naive/"
ANCE_MODEL_DIR=$TACO_DIR"/model/ance_hn_mt/mt_msmarco/naive/"
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
mkdir -p $ANCE_MODEL_DIR
#mkdir -p $ANCE_PROCESSED_DIR
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mt_sets=(msmarco zeshel fever nq)

SAVE_STEP=10000
EVAL_STEP=300

eval_delay=30
epoch=15
lr=5e-6
p_len=160
log_step=100
bsz=50
n_passages=4
infer_bsz=4096
mt_method="naive"
rands_ratio=0.5
n_gpu=8

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
  max_p_len=160
  n_passage=8
  if [ ${mt_set} == nq ]; then
    RAW_DIR=$DATA_DIR/kilt/${mt_set}/raw/
    PROCESSED_DIR=$DATA_DIR/kilt/${mt_set}/processed/bm25/
  elif [ ${mt_set} == fever ]; then
    RAW_DIR=$DATA_DIR/beir/${mt_set}/raw/
    PROCESSED_DIR=$DATA_DIR/beir/${mt_set}/processed/bm25/
  else
    RAW_DIR=$DATA_DIR/${mt_set}/raw/
    PROCESSED_DIR=$DATA_DIR/${mt_set}/processed/bm25/
  fi
  train_path="$delimiter"$PROCESSED_DIR/train.jsonl
  val_path="$delimiter"$PROCESSED_DIR/val.jsonl
  mt_train_paths+=${train_path}
  mt_eval_paths+=${val_path}
  max_q_lens+="$delimiter"$max_q_len
  max_p_lens+="$delimiter"$max_p_len
  task_names+="$delimiter"${mt_set^^}
  mt_n_passages+="$delimiter"$n_passages
done

cd $CODE_DIR
export PYTHONPATH=.


#echo "start warmup training ... "
#torchrun --nproc_per_node=$n_gpu --standalone --nnodes=1 src/taco/driver/train_mt.py \
#    --output_dir $MODEL_DIR  \
#    --model_name_or_path $PLM_DIR/t5-base-scaled  \
#    --do_train  \
#    --eval_delay $eval_delay \
#    --save_strategy epoch \
#    --evaluation_strategy epoch \
#    --logging_steps $log_step \
#    --mt_train_paths $mt_train_paths  \
#    --mt_eval_paths $mt_eval_paths \
#    --fp16  \
#    --per_device_train_batch_size $bsz  \
#    --mt_train_n_passages $mt_n_passages \
#    --learning_rate $lr  \
#    --q_max_lens $max_q_lens  \
#    --p_max_lens $max_p_lens \
#    --task_names $task_names \
#    --num_train_epochs $epoch  \
#    --logging_dir $LOG_DIR  \
#    --negatives_x_device True \
#    --remove_unused_columns False \
#    --overwrite_output_dir True \
#    --dataloader_num_workers 0 \
#    --multi_label False \
#    --in_batch_negatives True \
#    --pooling first \
#    --positive_passage_no_shuffle True \
#    --negative_passage_no_shuffle True \
#    --add_rand_negs False \
#    --encoder_only False \
#    --save_total_limit 2 \
#    --load_best_model_at_end True \
#    --metric_for_best_model loss \
#    --up_sample True \
#    --weight_method $mt_method \
#    --select_all True \
#    --multi_mix_temp 4.0 \
#    --add_query_task_prefix False \
#    --log_gnorm False \
#    --data_cache_dir $CACHE_DIR
#    --resume_from_checkpoint $MODEL_DIR/checkpoint-39040


for mt_set in ${mt_sets[@]}
do
  if [ ${mt_set} == zeshel ]; then
    max_q_len=128
  elif [ ${mt_set} == fever ]; then
    max_q_len=64
  else
    max_q_len=32
  fi
  if [ ${mt_set} == nq ]; then
    RAW_DIR=$DATA_DIR/kilt/${mt_set}/raw/
    PROCESSED_DIR=$DATA_DIR/kilt/${mt_set}/processed/bm25/
    ANCE_PROCESSED_DIR=$DATA_DIR/kilt/${mt_set}/processed/ance_hn_mt/mt_msmarco/naive/hn_iter_0/
  elif [ ${mt_set} == fever ]; then
    RAW_DIR=$DATA_DIR/beir/${mt_set}/raw/
    PROCESSED_DIR=$DATA_DIR/beir/${mt_set}/processed/bm25/
    ANCE_PROCESSED_DIR=$DATA_DIR/beir/${mt_set}/processed/ance_hn_mt/mt_msmarco/naive/hn_iter_0/
  else
    RAW_DIR=$DATA_DIR/${mt_set}/raw/
    PROCESSED_DIR=$DATA_DIR/${mt_set}/processed/bm25/
    ANCE_PROCESSED_DIR=$DATA_DIR/${mt_set}/processed/ance_hn_mt/mt_msmarco/naive/hn_iter_0/
  fi
  mkdir -p $ANCE_PROCESSED_DIR
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
  if [ ${mt_set} == fever ] || [ ${mt_set} == zeshel ]; then
    echo "building dev index for ${mt_set} "
  #  python src/taco/driver/build_index.py  \
    rm $EMBEDDING_DIR/embeddings.*
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
        --dataloader_num_workers 32 \
        --cache_dir $CACHE_DIR

    echo "retrieve dev data of ${mt_set} ... "
    if [ ! -d "$RESULT_DIR/${mt_set}" ]; then
        mkdir -p $RESULT_DIR/${mt_set}
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
        --trec_save_path $RESULT_DIR/${mt_set}/dev.trec \
        --dataloader_num_workers 0 \
        --cache_dir $CACHE_DIR \
        --split_retrieve \
        --use_gpu

    $EVAL_DIR/trec_eval -c -mRprec -mrecip_rank.10 -mrecall.64,100 $RAW_DIR/dev.qrel.trec $RESULT_DIR/${mt_set}/dev.trec > $RESULT_DIR/${mt_set}/dev_results.txt
    if [ ${mt_set} == nq ]; then
      echo "page-level scoring ..."
      python scripts/kilt/convert_trec_to_provenance.py  \
        --trec_file $RESULT_DIR/${mt_set}/dev.trec  \
        --kilt_queries_file $RAW_DIR/${mt_set}-dev-kilt.jsonl  \
        --passage_collection $DATA_DIR/kilt/corpus/psgs_w100.tsv  \
        --output_provenance_file $RESULT_DIR/${mt_set}/provenance.json
      echo "get prediction file ... "
      python scripts/kilt/convert_to_evaluation.py \
        --kilt_queries_file $RAW_DIR/${mt_set}-dev-kilt.jsonl  \
        --provenance_file $RESULT_DIR/${mt_set}/provenance.json \
        --output_evaluation_file $RESULT_DIR/${mt_set}/preds.json
      echo "get scores ... "
      python scripts/kilt/evaluate_kilt.py $RESULT_DIR/${mt_set}/preds.json $RAW_DIR/${mt_set}-dev-kilt.jsonl \
        --ks 1,20,100 \
        --results_file $RESULT_DIR/${mt_set}/page-level-results.json
    elif [ ${mt_set} == zeshel ]; then
      echo "build val hard negatives for zeshel"
  #    mkdir -p $ANCE_PROCESSED_DIR/${mt_set}/hn_iter_0
      python src/taco/dataset/build_hn.py  \
          --tokenizer_name $PLM_DIR/t5-base-scaled  \
          --hn_file $RESULT_DIR/${mt_set}/dev.trec \
          --qrels $RAW_DIR/dev.qrel.tsv \
          --queries $RAW_DIR/dev.query.txt \
          --collection $dev_corpus_path \
          --save_to $ANCE_PROCESSED_DIR \
          --template "Title: <title> Text: <text>" \
          --add_rand_negs \
          --num_hards 32 \
          --num_rands 32 \
          --split dev \
          --seed 42 \
          --use_doc_id_map \
          --cache_dir $CACHE_DIR \
          --shuffle_negatives

    fi

    if [ ${mt_set} == zeshel ] || [ ${mt_set} == fever ]; then
      echo "evaluate test data ... "
      if [ ${mt_set} == zeshel ]; then
        echo "build test index for zeshel"
        rm $EMBEDDING_DIR/embeddings.*
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
            --dataloader_num_workers 32 \
            --cache_dir $CACHE_DIR
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
          --trec_save_path $RESULT_DIR/${mt_set}/test.trec \
          --dataloader_num_workers 0 \
          --cache_dir $CACHE_DIR \
          --split_retrieve \
          --use_gpu

      echo "evaluate test trec ... "
      $EVAL_DIR/trec_eval -c -mrecip_rank.10 -mrecall.64,100 $RAW_DIR/test.qrel.trec $RESULT_DIR/${mt_set}/test.trec > $RESULT_DIR/${mt_set}/test_results.txt

    fi
    echo "get preprocessed data of ${mt_set} for ance training"
    if [ ${mt_set} == zeshel ]; then
      echo "build train index for zeshel ... "
      rm $EMBEDDING_DIR/embeddings.*
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
        --dataloader_num_workers 32 \
        --cache_dir $CACHE_DIR
    fi
    echo "retrieving train ..."
  #    if [ ${mt_set} == msmarco ]; then
  #      echo "random down_sample msmarco"
  #      export RANDOM=42
  #      echo "random down_sample train queries ... "
  #      shuf -n 100000 $RAW_DIR/train.query.txt > $ANCE_PROCESSED_DIR/train.query.txt
  #      train_query_path=$ANCE_PROCESSED_DIR/train.query.txt
  #    else
    train_query_path=$RAW_DIR/train.query.txt
  #    fi
    python -m src.taco.driver.retrieve  \
        --output_dir $EMBEDDING_DIR/ \
        --model_name_or_path $MODEL_DIR \
        --per_device_eval_batch_size $infer_bsz  \
        --query_path $train_query_path  \
        --encoder_only False  \
        --query_template "<text>"  \
        --query_column_names  id,text \
        --q_max_len $max_q_len  \
        --fp16  \
        --trec_save_path $RESULT_DIR/${mt_set}/train.trec \
        --dataloader_num_workers 0 \
        --topk 100 \
        --cache_dir $CACHE_DIR \
        --split_retrieve \
        --use_gpu

    echo "building hard negatives of ance first episode for ${mt_set} ..."
  #  mkdir -p $ANCE_PROCESSED_DIR/${mt_set}/hn_iter_0
    if [ ${mt_set} == zeshel ]; then
      python src/taco/dataset/build_hn.py  \
        --tokenizer_name $PLM_DIR/t5-base-scaled  \
        --hn_file $RESULT_DIR/${mt_set}/train.trec \
        --qrels $RAW_DIR/train.qrel.tsv \
        --queries $train_query_path \
        --collection $train_corpus_path \
        --save_to $ANCE_PROCESSED_DIR \
        --template "Title: <title> Text: <text>" \
        --num_hards 32 \
        --num_rands 32 \
        --split train \
        --seed 42 \
        --cache_dir $CACHE_DIR \
        --use_doc_id_map \
        --shuffle_negatives
  #        --add_rand_negs \
    elif [ ${mt_set} == fever ]; then
       python src/taco/dataset/build_hn.py  \
        --tokenizer_name $PLM_DIR/t5-base-scaled  \
        --hn_file $RESULT_DIR/${mt_set}/train.trec \
        --qrels $RAW_DIR/train.qrel.tsv \
        --queries $train_query_path \
        --collection $train_corpus_path \
        --save_to $ANCE_PROCESSED_DIR \
        --template "Title: <title> Text: <text>" \
        --num_hards 32 \
        --num_rands 32 \
        --split train \
        --seed 42 \
        --cache_dir $CACHE_DIR \
        --shuffle_negatives
  #        --use_doc_id_map \
  #        --add_rand_negs \
    else
      python src/taco/dataset/build_hn.py  \
        --tokenizer_name $PLM_DIR/t5-base-scaled  \
        --hn_file $RESULT_DIR/${mt_set}/train.trec \
        --qrels $RAW_DIR/train.qrel.tsv \
        --queries $train_query_path \
        --collection $train_corpus_path \
        --save_to $ANCE_PROCESSED_DIR \
        --template "Title: <title> Text: <text>" \
        --num_hards 32 \
        --num_rands 32 \
        --split train \
        --seed 42 \
        --cache_dir $CACHE_DIR \
        --shuffle_negatives
  #        --use_doc_id_map \
  #        --add_rand_negs \
    fi


  #    echo "removing train ${mt_set} trec files"
  #    rm $RESULT_DIR/${mt_set}/train.trec

    echo "splitting ${mt_set} hn file"
    if [ ${mt_set} == zeshel ]; then
      mv $ANCE_PROCESSED_DIR/train_all.jsonl  $ANCE_PROCESSED_DIR/train.jsonl
      echo "splitting zeshel dev hn file"
      tail -n 500 $ANCE_PROCESSED_DIR/dev_all.jsonl > $ANCE_PROCESSED_DIR/val.jsonl
    else
      tail -n 500 $ANCE_PROCESSED_DIR/train_all.jsonl > $ANCE_PROCESSED_DIR/val.jsonl
      head -n -500 $ANCE_PROCESSED_DIR/train_all.jsonl > $ANCE_PROCESSED_DIR/train.jsonl
      rm $ANCE_PROCESSED_DIR/train_all.jsonl
    fi
  fi
done

#echo "copy warmed up model to ance iter 0 model folder"
#cp -r $MODEL_DIR  $ANCE_MODEL_DIR/hn_iter_0
echo "deleting warmed up embeddings ... "
rm $EMBEDDING_DIR/embeddings.corpus.rank.*
