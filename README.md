# TACO
Official implementation of TACO paper
# Support features:
- normal dense retrieval
    - hard negative mining: unify ANCE with Hard-nce style, the only
     difference is optimizer state
    - multi-label retrieval (support custom loss in loss.py)
- multi-task retrieval
    - PCG
    - CGD
    - GradNorm
    - TACO
    - Naive
    - split query encoder
    - adapter?
- standard reranker
- standard reader (extractive and generative)
# features to add
- generative retrieval
- atlas style distillation
- discrete retrieval (mmi, vqvae can be combined with generative retrieval or
 dense retrieval)
# data structure
- only save dev/test trec/results files for different hn iteration
- save train_hn files for different hn iteration
- overwrite embeddings for different hn iteration
- each iteration save a single model or save a single model for different hn
 iteration (hard-nce style)
```
$DATA_DIR
    ├── data
    │    ├── kilt
    │    │   ├── raw
    │    │   │   ├── tasks queries and qrels
    │    │   │   └── corpus
    │    │   └── processed
    │    │       ├── bm25
    │    │       │   └── tasks
    │    │       ├── mt
    │    │       │   └── mt_methods
                         ├── add_prefix
                             └── hn_iters
                                 └── tasks
                         └── no_prefix   
    │    │       │           └── hn_iters
                                 └── tasks
    │    │       └── dr
                     └── tasks
                         └── hn_iters
    │    ├── msmarco
    │    ├── beir
    │    ├── dpr
    │    ├── zeshel
    │    └── ent_qa
    ├── plm
    ├── model
    ├── embeddings
    ├── trec_predicts
    ├── results
    ├── logs
    ├── metrics
    │   ├── __init__.py
    │   ├── beir
    │   ├── evaluate.sh
    │   └── trec

```
 
# TODOs:

- ~~Day1: dataset and modeling part~~
- ~~Day2: driver, retriever and trainer part~~
    - ~~retriever support data parallel and ddp~~
    - ~~trainer modify train loop to support hard negative mining instead of
     ance pipeline in scripts~~
    - ~~multi-task algorithms in mt_trainer~~
    - ~~standard dense_trainer~~
- Day3: scripts
- Day4: reranker and extractive reader part
- Day5: generative retrieval (CorpusBrain, Genre)
- Day6: discrete dense retrieval

# Install Faiss-gpu
For A100 CUDA-11, we need to install faiss-gpu via conda instead of pip

```
# check your cudatoolkit
conda search cudatoolkit
# install faiss-gpu 
conda install -c pytorch faiss-gpu cudatoolkit=11.3.1
```
