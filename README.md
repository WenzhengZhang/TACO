# TACO
Official implementation of TACO paper. Our code is built on 
[Openmatch](https://github.com/OpenMatch/OpenMatch).

# Setup
```
pip install -r requirements.txt
pip install -e .

```

# Install Faiss-gpu
For A100 CUDA-11, we need to install faiss-gpu via conda or specific pip wheels

- via conda

```
# check your cudatoolkit
conda search cudatoolkit
# install faiss-gpu 
conda install -c pytorch faiss-gpu cudatoolkit=11.3.1
```

- via pip
```
# download faiss-gpu pip cuda-11 wheel
wget https://github.com/kyamagu/faiss-wheels/releases/download/v1.7.3/faiss_gpu-1.7.3-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# pip install
python -m pip install faiss_gpu-1.7.3-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

```

# Supported Datasets and Setup DATA
- supported datasets
    - KILT Benchmark
    - MSMARCO
    - ZESHEL
    - BEIR
- setup data scripts (kilt for example)

```

./scripts/kilt/shells/setup_data.sh

```
For ZESHEL, you need to download the raw data and bm25 candidates from [here
](https://github.com/lajanugen/zeshel), then run the setup_data.sh script



# Supported features:
- standard dense retrieval
    - hard negative mining: unify ANCE with Hard-nce style, the only
     difference is optimizer state
- multi-task retrieval
    - PCG
    - CGD
    - GradNorm
    - TACO
    - Naive
    - split query encoder
    - query adapter

# example Scripts

- T5-ANCE dense retrieval

```
# warmup with bm25 candidates first
./scripts/kilt/shells/warmup_dr.sh

# ANCE iterations
./scripts/kilt/shells/ance_dr.sh

```

- Multi_task dense retrieval

```
# warmup with bm25 candidates first
./scripts/kilt/shells/warmup_mt.sh 

# ANCE iterations
./scripts/kilt/shells/ance_mt.sh [your multi_task method (naive, pcg, gn, cgd
, taco]

```


