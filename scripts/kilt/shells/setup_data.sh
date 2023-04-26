#!/bin/bash

HOME_DIR="/data/t-vincent" ### dear user: change this line and only this line
CODE_DIR=$HOME_DIR"/code/UnivSearchDev/"
PLM_DIR=$HOME_DIR"/model_checkpoints"
COLLECTION_DIR=$HOME_DIR"/data/kilt"
PROCESSED_DIR=$HOME_DIR"/data/kilt/processed_data"
TENSORBOARD_DIR="$HOME_DIR/tensorboard"
LOG_DIR="$HOME_DIR/logs"
CHECKPOINT_DIR=$HOME_DIR"/model_checkpoints"
EMBEDDING_DIR=$HOME_DIR"/embeddings_cache"
RESULT_DIR=$HOME_DIR"/result"

mkdir -p $PLM_DIR
mkdir -p $COLLECTION_DIR
mkdir -p $PROCESSED_DIR
mkdir $LOG_DIR
mkdir $TENSORBOARD_DIR
mkdir $CHECKPOINT_DIR
mkdir $EMBEDDING_DIR
mkdir $RESULT_DIR
mkdir -p $RESULT_DIR/kilt/
mkdir -p $PROCESSED_DIR/kilt/
mkdir -p $EMBEDDING_DIR/kilt/

### download pretrained language model, in this case T5
if [ -d "$PLM_DIR/t5-base-scaled" ]; then
    echo "$PLM_DIR/t5-base-scaled already exists.";
else
    echo "downloading T5 checkpoint into $PLM_DIR/t5-base-scaled";
    python $CODE_DIR/lib/scripts/scale_t5_weights.py \
        --input_model_path $PLM_DIR/t5-base \
        --output_model_path $PLM_DIR/t5-base-scaled \
        --model_name_or_path t5-base\
        --num_layers 12

fi
### download and process data

cd $COLLECTION_DIR
mkdir -p raw_data
mkdir -p raw_data/nq
mkdir -p raw_data/tqa
mkdir -p raw_data/hopo
mkdir -p raw_data/wow
mkdir -p raw_data/trex
mkdir -p raw_data/fever
mkdir -p raw_data/zsre
mkdir -p raw_data/aida
mkdir -p raw_data/corpus
echo "downloading kilt data"
if [ ! -f "$COLLECTION_DIR/raw_data/kilt_wikipedia_split/psgs_w100.tsv" ]; then
    python $CODE_DIR/lib/scripts/download_data.py --resource data.kilt_wikipedia_split.psgs_w100
    mv downloads/data/kilt_wikipedia_split raw_data/
fi
if [ ! -f "$COLLECTION_DIR/raw_data/nq/kilt-nq-train.json" ]; then
    python $CODE_DIR/lib/scripts/download_data.py --resource data.retriever.kilt-nq-train
    python $CODE_DIR/lib/scripts/download_data.py --resource data.retriever.kilt-nq-dev
    mv downloads/data/retriever/kilt-nq-train.json raw_data/nq/
    mv downloads/data/retriever/kilt-nq-dev.json raw_data/nq/
fi
if [ ! -f "$COLLECTION_DIR/raw_data/nq/nq-dev-kilt.jsonl" ]; then
    cd raw_data/nq
    wget http://dl.fbaipublicfiles.com/KILT/nq-dev-kilt.jsonl
    cd $COLLECTION_DIR
fi
if [ ! -f "$COLLECTION_DIR/raw_data/tqa/kilt-trivaqa-train.json" ]; then
    python $CODE_DIR/lib/scripts/download_data.py --resource data.retriever.kilt-trivaqa-train
    python $CODE_DIR/lib/scripts/download_data.py --resource data.retriever.kilt-trivaqa-dev
    mv downloads/data/retriever/kilt-trivaqa-train.json raw_data/tqa/
    mv downloads/data/retriever/kilt-trivaqa-dev.json raw_data/tqa/
fi
if [ ! -f "$COLLECTION_DIR/raw_data/tqa/tqa-dev-kilt.jsonl" ]; then
    cd raw_data/tqa
    wget http://dl.fbaipublicfiles.com/KILT/triviaqa-dev_id-kilt.jsonl
    python $CODE_DIR/lib/scripts/kilt/get_kilt_tqa_raw.py --base $COLLECTION_DIR/raw_data/tqa/
    cd $COLLECTION_DIR
fi
if [ ! -f "$COLLECTION_DIR/raw_data/hopo/kilt-hopo-train.json" ]; then
    python $CODE_DIR/lib/scripts/download_data.py --resource data.retriever.kilt-hopo-train
    python $CODE_DIR/lib/scripts/download_data.py --resource data.retriever.kilt-hopo-dev
    mv downloads/data/retriever/kilt-hopo-train.json raw_data/hopo/
    mv downloads/data/retriever/kilt-hopo-dev.json raw_data/hopo/
fi
if [ ! -f "$COLLECTION_DIR/raw_data/hopo/hopo-dev-kilt.jsonl" ]; then
    cd raw_data/hopo
    wget http://dl.fbaipublicfiles.com/KILT/hotpotqa-dev-kilt.jsonl
    mv hotpotqa-dev-kilt.jsonl hopo-dev-kilt.jsonl
    cd $COLLECTION_DIR
fi
if [ ! -f "$COLLECTION_DIR/raw_data/wow/kilt-wow-train.json" ]; then
    python $CODE_DIR/lib/scripts/download_data.py --resource data.retriever.kilt-wow-train
    python $CODE_DIR/lib/scripts/download_data.py --resource data.retriever.kilt-wow-dev
    mv downloads/data/retriever/kilt-wow-train.json raw_data/wow/
    mv downloads/data/retriever/kilt-wow-dev.json raw_data/wow/
fi
if [ ! -f "$COLLECTION_DIR/raw_data/wow/wow-dev-kilt.jsonl" ]; then
    cd raw_data/wow
    wget http://dl.fbaipublicfiles.com/KILT/wow-dev-kilt.jsonl
    cd $COLLECTION_DIR
fi
if [ ! -f "$COLLECTION_DIR/raw_data/trex/kilt-trex-train.json" ]; then
    python $CODE_DIR/lib/scripts/download_data.py --resource data.retriever.kilt-trex-train
    python $CODE_DIR/lib/scripts/download_data.py --resource data.retriever.kilt-trex-dev
    mv downloads/data/retriever/kilt-trex-train.json raw_data/trex/
    mv downloads/data/retriever/kilt-trex-dev.json raw_data/trex/
fi
if [ ! -f "$COLLECTION_DIR/raw_data/trex/trex-dev-kilt.jsonl" ]; then
    cd raw_data/trex
    wget http://dl.fbaipublicfiles.com/KILT/trex-dev-kilt.jsonl
    cd $COLLECTION_DIR
fi
if [ ! -f "$COLLECTION_DIR/raw_data/fever/kilt-fever-train.json" ]; then
    python $CODE_DIR/lib/scripts/download_data.py --resource data.retriever.kilt-fever-train
    python $CODE_DIR/lib/scripts/download_data.py --resource data.retriever.kilt-fever-dev
    mv downloads/data/retriever/kilt-fever-train.json raw_data/fever/
    mv downloads/data/retriever/kilt-fever-dev.json raw_data/fever/
fi
if [ ! -f "$COLLECTION_DIR/raw_data/fever/fever-dev-kilt.jsonl" ]; then
    cd raw_data/fever
    wget http://dl.fbaipublicfiles.com/KILT/fever-dev-kilt.jsonl
    cd $COLLECTION_DIR
fi
if [ ! -f "$COLLECTION_DIR/raw_data/zsre/kilt-zeroshot-train.json" ]; then
    python $CODE_DIR/lib/scripts/download_data.py --resource data.retriever.kilt-zeroshot-train
    python $CODE_DIR/lib/scripts/download_data.py --resource data.retriever.kilt-zeroshot-dev
    mv downloads/data/retriever/kilt-zeroshot-train.json raw_data/zsre/
    mv downloads/data/retriever/kilt-zeroshot-dev.json raw_data/zsre/
fi
if [ ! -f "$COLLECTION_DIR/raw_data/zsre/zsre-dev-kilt.jsonl" ]; then
    cd raw_data/zsre
    wget http://dl.fbaipublicfiles.com/KILT/structured_zeroshot-dev-kilt.jsonl
    mv structured_zeroshot-dev-kilt.jsonl zsre-dev-kilt.jsonl
    cd $COLLECTION_DIR
fi
if [ ! -f "$COLLECTION_DIR/raw_data/aida/kilt-aida-train.json" ]; then
    python $CODE_DIR/lib/scripts/download_data.py --resource data.retriever.kilt-aida-train
    mv downloads/data/retriever/kilt-aida-train.json raw_data/aida/
fi
if [ ! -f "$COLLECTION_DIR/raw_data/aida/aida-dev-kilt.jsonl" ]; then
    cd raw_data/aida
    wget http://dl.fbaipublicfiles.com/KILT/aidayago2-dev-kilt.jsonl
    mv aidayago2-dev-kilt.jsonl aida-dev-kilt.jsonl
    cd $COLLECTION_DIR
fi

echo "processing kilt data ... "
echo "process nq ... "
if [ ! -f "$COLLECTION_DIR/raw_data/nq/train-nq-query.txt" ]; then
    python $CODE_DIR/lib/scripts/kilt/process_kilt.py \
      --train_nq_file $COLLECTION_DIR/raw_data/nq/kilt-nq-train.json \
      --train_nq_query $COLLECTION_DIR/raw_data/nq/train-nq.query.txt \
      --train_nq_qrel $COLLECTION_DIR/raw_data/nq/train-nq.qrel.tsv \
      --dev_nq_file $COLLECTION_DIR/raw_data/nq/kilt-nq-dev.json \
      --dev_nq_query $COLLECTION_DIR/raw_data/nq/dev-nq.query.txt \
      --dev_nq_qrel $COLLECTION_DIR/raw_data/nq/dev-nq.qrel.trec \
      --process_nq
fi

echo "process tqa ... "
if [ ! -f "$COLLECTION_DIR/raw_data/tqa/train-tqa-query.txt" ]; then
    python $CODE_DIR/lib/scripts/kilt/process_kilt.py \
      --train_tqa_file $COLLECTION_DIR/raw_data/tqa/kilt-trivaqa-train.json \
      --train_tqa_query $COLLECTION_DIR/raw_data/tqa/train-tqa.query.txt \
      --train_tqa_qrel $COLLECTION_DIR/raw_data/tqa/train-tqa.qrel.tsv \
      --dev_tqa_file $COLLECTION_DIR/raw_data/tqa/kilt-trivaqa-dev.json \
      --dev_tqa_query $COLLECTION_DIR/raw_data/tqa/dev-tqa.query.txt \
      --dev_tqa_qrel $COLLECTION_DIR/raw_data/tqa/dev-tqa.qrel.trec \
      --process_tqa
fi

echo "process hopo ... "
if [ ! -f "$COLLECTION_DIR/raw_data/hopo/train-hopo-query.txt" ]; then
    python $CODE_DIR/lib/scripts/kilt/process_kilt.py \
      --train_hopo_file $COLLECTION_DIR/raw_data/hopo/kilt-hopo-train.json \
      --train_hopo_query $COLLECTION_DIR/raw_data/hopo/train-hopo.query.txt \
      --train_hopo_qrel $COLLECTION_DIR/raw_data/hopo/train-hopo.qrel.tsv \
      --dev_hopo_file $COLLECTION_DIR/raw_data/hopo/kilt-hopo-dev.json \
      --dev_hopo_query $COLLECTION_DIR/raw_data/hopo/dev-hopo.query.txt \
      --dev_hopo_qrel $COLLECTION_DIR/raw_data/hopo/dev-hopo.qrel.trec \
      --process_hopo
fi

echo "process wow ... "
if [ ! -f "$COLLECTION_DIR/raw_data/wow/train-wow-query.txt" ]; then
    python $CODE_DIR/lib/scripts/kilt/process_kilt.py \
      --train_wow_file $COLLECTION_DIR/raw_data/wow/kilt-wow-train.json \
      --train_wow_query $COLLECTION_DIR/raw_data/wow/train-wow.query.txt \
      --train_wow_qrel $COLLECTION_DIR/raw_data/wow/train-wow.qrel.tsv \
      --dev_wow_file $COLLECTION_DIR/raw_data/wow/kilt-wow-dev.json \
      --raw_dev_wow_file $COLLECTION_DIR/raw_data/wow/wow-dev-kilt.jsonl \
      --out_dev_wow_file $COLLECTION_DIR/raw_data/wow/kilt-wow-dev-orig.json \
      --dev_wow_query $COLLECTION_DIR/raw_data/wow/dev-wow.query.txt \
      --dev_wow_qrel $COLLECTION_DIR/raw_data/wow/dev-wow.qrel.trec \
      --process_wow
fi

echo "process trex ... "
if [ ! -f "$COLLECTION_DIR/raw_data/trex/train-trex-query.txt" ]; then
    python $CODE_DIR/lib/scripts/kilt/process_kilt.py \
      --train_trex_file $COLLECTION_DIR/raw_data/trex/kilt-trex-train.json \
      --train_trex_small $COLLECTION_DIR/raw_data/trex/kilt-trex-train-small.json \
      --train_trex_query $COLLECTION_DIR/raw_data/trex/train-trex.query.txt \
      --train_trex_qrel $COLLECTION_DIR/raw_data/trex/train-trex.qrel.tsv \
      --dev_trex_file $COLLECTION_DIR/raw_data/trex/kilt-trex-dev.json \
      --dev_trex_query $COLLECTION_DIR/raw_data/trex/dev-trex.query.txt \
      --dev_trex_qrel $COLLECTION_DIR/raw_data/trex/dev-trex.qrel.trec \
      --process_trex
fi

echo "process fever ... "
if [ ! -f "$COLLECTION_DIR/raw_data/fever/train-fever-query.txt" ]; then
    python $CODE_DIR/lib/scripts/kilt/process_kilt.py \
      --train_fever_file $COLLECTION_DIR/raw_data/fever/kilt-fever-train.json \
      --train_fever_query $COLLECTION_DIR/raw_data/fever/train-fever.query.txt \
      --train_fever_qrel $COLLECTION_DIR/raw_data/fever/train-fever.qrel.tsv \
      --dev_fever_file $COLLECTION_DIR/raw_data/fever/kilt-fever-dev.json \
      --dev_fever_query $COLLECTION_DIR/raw_data/fever/dev-fever.query.txt \
      --dev_fever_qrel $COLLECTION_DIR/raw_data/fever/dev-fever.qrel.trec \
      --process_fever
fi

echo "process zsre ... "
if [ ! -f "$COLLECTION_DIR/raw_data/zsre/train-zsre-query.txt" ]; then
    python $CODE_DIR/lib/scripts/kilt/process_kilt.py \
      --train_zsre_file $COLLECTION_DIR/raw_data/zsre/kilt-zeroshot-train.json \
      --train_zsre_small $COLLECTION_DIR/raw_data/zsre/kilt-zsre-train-small.json \
      --train_zsre_query $COLLECTION_DIR/raw_data/zsre/train-zsre.query.txt \
      --train_zsre_qrel $COLLECTION_DIR/raw_data/zsre/train-zsre.qrel.tsv \
      --dev_zsre_file $COLLECTION_DIR/raw_data/zsre/kilt-zeroshot-dev.json \
      --dev_zsre_query $COLLECTION_DIR/raw_data/zsre/dev-zsre.query.txt \
      --dev_zsre_qrel $COLLECTION_DIR/raw_data/zsre/dev-zsre.qrel.trec \
      --process_zsre
fi

echo "process aida ... "
if [ ! -f "$COLLECTION_DIR/raw_data/aida/train-aida-query.txt" ]; then
    python $CODE_DIR/lib/scripts/kilt/process_kilt.py \
      --train_aida_file $COLLECTION_DIR/raw_data/aida/kilt-aida-train.json \
      --train_aida_out $COLLECTION_DIR/raw_data/aida/aida-train.json \
      --train_aida_query $COLLECTION_DIR/raw_data/aida/train-aida.query.txt \
      --train_aida_qrel $COLLECTION_DIR/raw_data/aida/train-aida.qrel.tsv \
      --dev_aida_file $COLLECTION_DIR/raw_data/aida/aida-dev-kilt.jsonl \
      --dev_aida_query $COLLECTION_DIR/raw_data/aida/dev-aida.query.txt \
      --dev_aida_qrel $COLLECTION_DIR/raw_data/aida/dev-aida.qrel.trec \
      --kilt_psgs $COLLECTION_DIR/raw_data/kilt_wikipedia_split/psgs_w100.tsv \
      --process_aida
fi

echo "process psg corpus ... "
if [ ! -f "$COLLECTION_DIR/raw_data/corpus/psg_corpus.tsv" ]; then
    python $CODE_DIR/lib/scripts/kilt/process_kilt.py \
      --kilt_psgs $COLLECTION_DIR/raw_data/kilt_wikipedia_split/psgs_w100.tsv \
      --kilt_psg_corpus $COLLECTION_DIR/raw_data/corpus/psg_corpus.tsv \
      --process_psgs
fi

if [ -d "$COLLECTION_DIR/downloads" ]; then
    rm -rf $COLLECTION_DIR/downloads
fi

### build kilt train data
echo "build kilt train data"
cd $CODE_DIR

export PYTHONPATH=.

echo "build nq train ... "

if [ -f "$PROCESSED_DIR/nq/nq-train.new.jsonl" ]; then
    echo "$PROCESSED_DIR/nq/nq-train.new.jsonl exists";
else
    echo "RUNNING build_train.py...";
    echo "building training data ... "
    python $CODE_DIR/lib/scripts/kilt/build_train.py \
        --tokenizer $PLM_DIR/t5-base-scaled  \
        --input $COLLECTION_DIR/raw_data/nq/kilt-nq-train.json \
        --output $PROCESSED_DIR/nq/nq-train.jsonl \
        --minimum-negatives 1 \
        --template "Title: <title> Text: <text>" \
        --max_q_len 32 \
        --max_length 160
    tail -500 $PROCESSED_DIR/nq/nq-train.jsonl > $PROCESSED_DIR/nq/nq-val.jsonl
    head -76445 $PROCESSED_DIR/nq/nq-train.jsonl > $PROCESSED_DIR/nq/nq-train.new.jsonl

fi

echo "build tqa train ... "

if [ -f "$PROCESSED_DIR/tqa/tqa-train.new.jsonl" ]; then
    echo "$PROCESSED_DIR/tqa/tqa-train.new.jsonl exists";
else
    echo "RUNNING build_train.py...";
    echo "building training data ... "
    python $CODE_DIR/lib/scripts/kilt/build_train.py \
        --tokenizer $PLM_DIR/t5-base-scaled  \
        --input $COLLECTION_DIR/raw_data/tqa/kilt-trivaqa-train.json \
        --output $PROCESSED_DIR/tqa/tqa-train.jsonl \
        --minimum-negatives 1 \
        --template "Title: <title> Text: <text>" \
        --max_q_len 32 \
        --max_length 160
    tail -500 $PROCESSED_DIR/tqa/tqa-train.jsonl > $PROCESSED_DIR/tqa/tqa-val.jsonl
    head -52386 $PROCESSED_DIR/tqa/tqa-train.jsonl > $PROCESSED_DIR/tqa/tqa-train.new.jsonl

fi

echo "build hopo train ... "

if [ -f "$PROCESSED_DIR/hopo/hopo-train.new.jsonl" ]; then
    echo "$PROCESSED_DIR/hopo/hopo-train.new.jsonl exists";
else
    echo "RUNNING build_train.py...";
    echo "building training data ... "
    python $CODE_DIR/lib/scripts/kilt/build_train.py \
        --tokenizer $PLM_DIR/t5-base-scaled  \
        --input $COLLECTION_DIR/raw_data/hopo/kilt-hopo-train.json \
        --output $PROCESSED_DIR/hopo/hopo-train.jsonl \
        --minimum-negatives 1 \
        --template "Title: <title> Text: <text>" \
        --max_q_len 32 \
        --max_length 160
    tail -500 $PROCESSED_DIR/hopo/hopo-train.jsonl > $PROCESSED_DIR/hopo/hopo-val.jsonl
    head -68159 $PROCESSED_DIR/hopo/hopo-train.jsonl > $PROCESSED_DIR/hopo/hopo-train.new.jsonl

fi

echo "build wow train ... "

if [ -f "$PROCESSED_DIR/wow/wow-train.new.jsonl" ]; then
    echo "$PROCESSED_DIR/wow/wow-train.new.jsonl exists";
else
    echo "RUNNING build_train.py...";
    echo "building training data ... "
    python $CODE_DIR/lib/scripts/kilt/build_train.py \
        --tokenizer $PLM_DIR/t5-base-scaled  \
        --input $COLLECTION_DIR/raw_data/wow/kilt-wow-train.json \
        --output $PROCESSED_DIR/wow/wow-train.jsonl \
        --minimum-negatives 1 \
        --template "Title: <title> Text: <text>" \
        --max_q_len 256 \
        --max_length 160
    tail -500 $PROCESSED_DIR/wow/wow-train.jsonl > $PROCESSED_DIR/wow/wow-val.jsonl
    head -79535 $PROCESSED_DIR/wow/wow-train.jsonl > $PROCESSED_DIR/wow/wow-train.new.jsonl

fi

echo "build trex train ... "

if [ -f "$PROCESSED_DIR/trex/trex-train.new.jsonl" ]; then
    echo "$PROCESSED_DIR/trex/trex-train.new.jsonl exists";
else
    echo "RUNNING build_train.py...";
    echo "building training data ... "
    python $CODE_DIR/lib/scripts/kilt/build_train.py \
        --tokenizer $PLM_DIR/t5-base-scaled  \
        --input $COLLECTION_DIR/raw_data/trex/kilt-trex-train-small.json \
        --output $PROCESSED_DIR/trex/trex-train.jsonl \
        --minimum-negatives 1 \
        --template "Title: <title> Text: <text>" \
        --max_q_len 32 \
        --max_length 160
    tail -500 $PROCESSED_DIR/trex/trex-train.jsonl > $PROCESSED_DIR/trex/trex-val.jsonl
    head -94514 $PROCESSED_DIR/trex/trex-train.jsonl > $PROCESSED_DIR/trex/trex-train.new.jsonl

fi

echo "build fever train ... "

if [ -f "$PROCESSED_DIR/fever/fever-train.new.jsonl" ]; then
    echo "$PROCESSED_DIR/fever/fever-train.new.jsonl exists";
else
    echo "RUNNING build_train.py...";
    echo "building training data ... "
    python $CODE_DIR/lib/scripts/kilt/build_train.py \
        --tokenizer $PLM_DIR/t5-base-scaled  \
        --input $COLLECTION_DIR/raw_data/fever/kilt-fever-train.json \
        --output $PROCESSED_DIR/fever/fever-train.jsonl \
        --minimum-negatives 1 \
        --template "Title: <title> Text: <text>" \
        --max_q_len 64 \
        --max_length 160
    tail -500 $PROCESSED_DIR/fever/fever-train.jsonl > $PROCESSED_DIR/fever/fever-val.jsonl
    head -70757 $PROCESSED_DIR/fever/fever-train.jsonl > $PROCESSED_DIR/fever/fever-train.new.jsonl

fi

echo "build zsre train ... "

if [ -f "$PROCESSED_DIR/zsre/zsre-train.new.jsonl" ]; then
    echo "$PROCESSED_DIR/zsre/zsre-train.new.jsonl exists";
else
    echo "RUNNING build_train.py...";
    echo "building training data ... "
    python $CODE_DIR/lib/scripts/kilt/build_train.py \
        --tokenizer $PLM_DIR/t5-base-scaled  \
        --input $COLLECTION_DIR/raw_data/zsre/kilt-zsre-train-small.json \
        --output $PROCESSED_DIR/zsre/zsre-train.jsonl \
        --minimum-negatives 1 \
        --template "Title: <title> Text: <text>" \
        --max_q_len 32 \
        --max_length 160
    tail -500 $PROCESSED_DIR/zsre/zsre-train.jsonl > $PROCESSED_DIR/zsre/zsre-val.jsonl
    head -99500 $PROCESSED_DIR/zsre/zsre-train.jsonl > $PROCESSED_DIR/zsre/zsre-train.new.jsonl

fi

echo "build aida train ... "

if [ -f "$PROCESSED_DIR/aida/aida-train.new.jsonl" ]; then
    echo "$PROCESSED_DIR/aida/aida-train.new.jsonl exists";
else
    echo "RUNNING build_train.py...";
    echo "building training data ... "
    python $CODE_DIR/lib/scripts/kilt/build_train.py \
        --tokenizer $PLM_DIR/t5-base-scaled  \
        --input $COLLECTION_DIR/raw_data/aida/aida-train.json \
        --output $PROCESSED_DIR/aida/aida-train.jsonl \
        --minimum-negatives 1 \
        --template "Title: <title> Text: <text>" \
        --max_q_len 128 \
        --max_length 160
    tail -500 $PROCESSED_DIR/aida/aida-train.jsonl > $PROCESSED_DIR/aida/aida-val.jsonl
    head -17895 $PROCESSED_DIR/aida/aida-train.jsonl > $PROCESSED_DIR/aida/aida-train.new.jsonl

fi

echo "Done setting up data and environments"
