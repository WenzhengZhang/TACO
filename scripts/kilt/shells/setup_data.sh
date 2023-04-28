#!/bin/bash

HOME_DIR="/common/users/wz283/projects/"
CODE_DIR=$HOME_DIR"/TACO/"
TACO_DIR=$HOME_DIR"/taco_data/"
PLM_DIR=$TACO_DIR"/plm/"
MODEL_DIR=$TACO_DIR"/model/kilt"
DATA_DIR=$TACO_DIR"/data/kilt/"
RAW_DIR=$DATA_DIR"/raw/"
PROCESSED_DIR=$DATA_DIR"/processed/"
EVAL_DIR=$TACO_DIR"/metrics/trec/trec_eval-9.0.7/"
mkdir -p $TACO_DIR
mkdir -p $PLM_DIR
mkdir -p $MODEL_DIR
mkdir -p $DATA_DIR
mkdir -p $RAW_DIR
mkdir -p $PROCESSED_DIR
mkdir -p $EVAL_DIR
kilt_sets=(nq tqa hopo wow trex fever zsre aida)
for dataset in ${kilt_sets[@]}
do
  mkdir -p $RAW_DIR/$dataset
  mkdir -p $PROCESSED_DIR/"bm25/"$dataset
done
mkdir -p $RAW_DIR/"corpus"


### download pretrained language model, in this case T5
if [ -d "$PLM_DIR/t5-base-scaled" ]; then
    echo "$PLM_DIR/t5-base-scaled already exists.";
else
    echo "downloading T5 checkpoint into $PLM_DIR/t5-base-scaled";
    python $CODE_DIR/scripts/scale_t5_weights.py \
        --input_model_path $PLM_DIR/t5-base \
        --output_model_path $PLM_DIR/t5-base-scaled \
        --model_name_or_path t5-base \
        --num_layers 12

fi
### download and process data
cd $RAW_DIR

echo "downloading kilt data"
if [ ! -f "$RAW_DIR/corpus/psgs_w100.tsv" ]; then
    python $CODE_DIR/scripts/download_data.py --resource data.kilt_wikipedia_split.psgs_w100
    mv downloads/data/kilt_wikipedia_split/psgs_w100.tsv "$RAW_DIR/corpus/"
    rm -rf downloads
fi
echo "downloading kilt datasets"
for kilt_set in ${kilt_sets[@]}
do
  if [ $kilt_set == tqa ]; then
    download_name="triviaqa"
  elif [ $kilt_set == zsre ]; then
    download_name="zeroshot"
  else
    download_name=$kilt_set
  fi
  if [ ! -f "$RAW_DIR/${kilt_set}/train.json" ]; then
    python $CODE_DIR/scripts/download_data.py --resource data.retriever.kilt-${download_name}-train
    if [ $download_name != aida ]; then
      python $CODE_DIR/scripts/download_data.py --resource data.retriever.kilt-${download_name}-dev
      mv downloads/data/retriever/kilt-${download_name}-dev.json "$RAW_DIR/$kilt_set/dev.json"
    fi
    mv downloads/data/retriever/kilt-${download_name}-train.json "$RAW_DIR/$kilt_set/train.json"
    rm -rf downloads
  fi
  cd "$RAW_DIR/${kilt_set}"
  if [ $kilt_set == tqa ]; then
    kilt_orig_dev="triviaqa-dev_id-kilt.jsonl"
    wget http://dl.fbaipublicfiles.com/KILT/${kilt_orig_dev}
    python $CODE_DIR/scripts/kilt/get_kilt_tqa_raw.py --base "$RAW_DIR/"
  elif [ $kilt_set == hopo ]; then
    kilt_orig_dev="hotpotqa-dev-kilt.jsonl"
    wget http://dl.fbaipublicfiles.com/KILT/${kilt_orig_dev}
    mv ${kilt_orig_dev} "${kilt_set}-dev-kilt.jsonl"
  elif [ $kilt_set == zsre ]; then
    kilt_orig_dev="structured_zeroshot-dev-kilt.jsonl"
    wget http://dl.fbaipublicfiles.com/KILT/${kilt_orig_dev}
    mv ${kilt_orig_dev} "${kilt_set}-dev-kilt.jsonl"
  elif [ $kilt_set == aida ]; then
    kilt_orig_dev="aidayago2-dev-kilt.jsonl"
    wget http://dl.fbaipublicfiles.com/KILT/${kilt_orig_dev}
    mv ${kilt_orig_dev} "${kilt_set}-dev-kilt.jsonl"
  else
    kilt_orig_dev="${kilt_set}-dev-kilt.jsonl"
    wget http://dl.fbaipublicfiles.com/KILT/${kilt_orig_dev}
  fi
  cd $RAW_DIR
done

echo "processing kilt data ... "
echo "process kilt sets  ... "
if [ ! -f "$RAW_DIR/tqa/train.query.txt" ]; then
    python $CODE_DIR/scripts/kilt/process_kilt.py \
      --train_nq_file $RAW_DIR/nq/train.json \
      --train_nq_query $RAW_DIR/nq/train.query.txt \
      --train_nq_qrel $RAW_DIR/nq/train.qrel.tsv \
      --dev_nq_file $RAW_DIR/nq/dev.json \
      --dev_nq_query $RAW_DIR/nq/dev.query.txt \
      --dev_nq_qrel $RAW_DIR/nq/dev.qrel.trec \
      --process_nq \
      --train_tqa_file $RAW_DIR/tqa/train.json \
      --train_tqa_query $RAW_DIR/tqa/train.query.txt \
      --train_tqa_qrel $RAW_DIR/tqa/train.qrel.tsv \
      --dev_tqa_file $RAW_DIR/tqa/dev.json \
      --dev_tqa_query $RAW_DIR/tqa/dev.query.txt \
      --dev_tqa_qrel $RAW_DIR/tqa/dev.qrel.trec \
      --process_tqa \
      --train_hopo_file $RAW_DIR/hopo/train.json \
      --train_hopo_query $RAW_DIR/hopo/train.query.txt \
      --train_hopo_qrel $RAW_DIR/hopo/train.qrel.tsv \
      --dev_hopo_file $RAW_DIR/hopo/dev.json \
      --dev_hopo_query $RAW_DIR/hopo/dev.query.txt \
      --dev_hopo_qrel $RAW_DIR/hopo/dev.qrel.trec \
      --process_hopo \
      --train_wow_file $RAW_DIR/wow/train.json \
      --train_wow_query $RAW_DIR/wow/train.query.txt \
      --train_wow_qrel $RAW_DIR/wow/train.qrel.tsv \
      --dev_wow_file $RAW_DIR/wow/dev.json \
      --raw_dev_wow_file $RAW_DIR/wow/wow-dev-kilt.jsonl \
      --out_dev_wow_file $RAW_DIR/wow/wow-dev.json \
      --dev_wow_query $RAW_DIR/wow/dev.query.txt \
      --dev_wow_qrel $RAW_DIR/wow/dev.qrel.trec \
      --process_wow \
      --train_trex_file $RAW_DIR/trex/train.json \
      --train_trex_small $RAW_DIR/trex/train-small.json \
      --train_trex_query $RAW_DIR/trex/train.query.txt \
      --train_trex_qrel $RAW_DIR/trex/train.qrel.tsv \
      --dev_trex_file $RAW_DIR/trex/dev.json \
      --dev_trex_query $RAW_DIR/trex/dev.query.txt \
      --dev_trex_qrel $RAW_DIR/trex/dev.qrel.trec \
      --process_trex \
      --train_fever_file $RAW_DIR/fever/train.json \
      --train_fever_query $RAW_DIR/fever/train.query.txt \
      --train_fever_qrel $RAW_DIR/fever/train.qrel.tsv \
      --dev_fever_file $RAW_DIR/fever/dev.json \
      --dev_fever_query $RAW_DIR/fever/dev.query.txt \
      --dev_fever_qrel $RAW_DIR/fever/dev.qrel.trec \
      --process_fever \
      --train_zsre_file $RAW_DIR/zsre/train.json \
      --train_zsre_small $RAW_DIR/zsre/train-small.json \
      --train_zsre_query $RAW_DIR/zsre/train.query.txt \
      --train_zsre_qrel $RAW_DIR/zsre/train.qrel.tsv \
      --dev_zsre_file $RAW_DIR/zsre/dev.json \
      --dev_zsre_query $RAW_DIR/zsre/dev.query.txt \
      --dev_zsre_qrel $RAW_DIR/zsre/dev.qrel.trec \
      --process_zsre \
      --train_aida_file $RAW_DIR/aida/train.json \
      --train_aida_out $RAW_DIR/aida/aida-train.json \
      --train_aida_query $RAW_DIR/aida/train.query.txt \
      --train_aida_qrel $RAW_DIR/aida/train.qrel.tsv \
      --dev_aida_file $RAW_DIR/aida/aida-dev-kilt.jsonl \
      --dev_aida_query $RAW_DIR/aida/dev.query.txt \
      --dev_aida_qrel $RAW_DIR/aida/dev.qrel.trec \
      --kilt_psgs $RAW_DIR/corpus/psgs_w100.tsv \
      --process_aida
fi

echo "process psg corpus ... "
if [ ! -f "$RAW_DIR/corpus/psg_corpus.tsv" ]; then
    python $CODE_DIR/scripts/kilt/process_kilt.py \
      --kilt_psgs $RAW_DIR/corpus/psgs_w100.tsv \
      --kilt_psg_corpus $RAW_DIR/corpus/psg_corpus.tsv \
      --process_psgs
fi

if [ -d "$RAW_DIR/downloads" ]; then
    rm -rf $RAW_DIR/downloads
fi

### build kilt train data
echo "build kilt train data"
cd $CODE_DIR

export PYTHONPATH=.
for kilt_set in ${kilt_sets[@]}
do
  echo "build ${kilt_set} train ..."
  max_d_len=160
  if [ ${kilt_set} == wow ]; then
    max_q_len=256
  elif [ ${kilt_set} == fever ]; then
    max_q_len=64
  elif [ ${kilt_set} == aida ]; then
    max_q_len=128
  else
    max_q_len=32
  fi
  if [ ${kilt_set} == trex ] || [ ${kilt_set} == zsre ]; then
    kilt_set_input=$RAW_DIR/${kilt_set}/train-small.json
  else
    kilt_set_input=$RAW_DIR/${kilt_set}/train.json
  fi
  if [ -f "$PROCESSED_DIR/"bm25/"${kilt_set}/train.jsonl" ]; then
    echo "$PROCESSED_DIR/"bm25/"${kilt_set}/train.jsonl";
  else
    echo "RUNNING build_train.py...";
    echo "building training data ... "
    python $CODE_DIR/scripts/kilt/build_train.py \
        --tokenizer $PLM_DIR/t5-base-scaled  \
        --input ${kilt_set_input} \
        --output $PROCESSED_DIR/"bm25/"${kilt_set}/train_all.jsonl \
        --minimum-negatives 1 \
        --template "Title: <title> Text: <text>" \
        --max_q_len $max_q_len \
        --max_length $max_d_len
    tail -n 500 $PROCESSED_DIR/"bm25/"${kilt_set}/train_all.jsonl > $PROCESSED_DIR/"bm25/"${kilt_set}/val.jsonl
    head -n -500 $PROCESSED_DIR/"bm25/"${kilt_set}/train_all.jsonl > $PROCESSED_DIR/"bm25/"${kilt_set}/train.jsonl
  fi
done
echo "Done setting up data and environments"
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