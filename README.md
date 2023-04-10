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
 
# TODOs:

- ~~Day1: dataset and modeling part~~
- Day2: driver, retriever and trainer part
    - retriever support data parallel and ddp
    - trainer modify train loop to support hard negative mining instead of
     ance pipeline in scripts
    - multi-task algorithms in mt_trainer
    - standard dense_trainer
- Day3: scripts
- Day4: reranker and extractive reader part
- Day5: generative retrieval (CorpusBrain, Genre)
- Day6: discrete dense retrieval
