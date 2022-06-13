# Notes on how to build Megatron-LM and its depednecies for NVIDIA GPUs

This is the process Sam's followed to get Megatron (NVIDIA) work on top ov NVHPC 21.3. 

This was done in a AMD internal machine that runs A100, here the details:
```
NVIDIA-SMI 465.19.01    Driver Version: 465.19.01    CUDA Version: 11.3
```

## Build 
Note that this machines have a LMOD hierarchy that provides the CUDA/NVHPC environment. 
Build is prefixed to `/home/sfantao/lumi-builds/megatron` but that could be adjusted.
Sam also chose to get cuDNN from RHEL8 distro as it matches the machine he was running on.

```
#
# Modules to provide CUDA and recent GCC if needed
#
ml nvhpc/21.3 #gcc/9.3.0 

#
# Environment with base dependencies
#
source ~/miniconda3/bin/activate
conda create -n megatron-base python=3.8
conda activate megatron-base

#
# Obtain cuDNN - running in RHEL
#
curl -LO https://developer.nvidia.com/compute/cudnn/secure/8.4.0/local_installers/11.6/cudnn-local-repo-rhel8-8.4.0.27-1.0-1.x86_64.rpm


#
# Checkout pytorch
#
cd $base
rm -rf pytorch
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout -b v1.8.1 v1.8.1
git submodule sync
git submodule update --init --recursive --jobs 0
cd -

#
# Install base dependencies
#
conda install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
conda install -c pytorch magma-cuda110 

#
# Build pytorch
#
export CMAKE_PREFIX_PATH=/home/sfantao/lumi-builds/megatron/miniconda3/envs/megatron:/share/modules/hpc_sdk/21.3/Linux_x86_64/21.3/math_libs/11.2/targets/x86_64-linux
cd $base
rm -rf build ; \
MKLROOT=/home/sfantao/lumi-builds/megatron/miniconda3/envs/megatron \
CC=gcc \
CXX=g++ \
USE_CUDA=1 \
USE_SYSTEM_NCCL=1 \
USE_NCCL=1 \
USE_CUDNN=1 \
BLAS=MKL \
TORCH_CUDA_ARCH_LIST="8.0" \
CUDA_HOME=$(realpath $(dirname $(which ptxas))/../) \
CUDNN_LIB_DIR=/home/sfantao/lumi-builds/megatron/deps/cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive/lib \
CUDNN_INCLUDE_DIR=/home/sfantao/lumi-builds/megatron/deps/cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive/include \
NCCL_INCLUDE_DIR=/share/modules/hpc_sdk/21.3/Linux_x86_64/21.3/comm_libs/nccl/include \
NCCL_LIB_DIR=/share/modules/hpc_sdk/21.3/Linux_x86_64/21.3/comm_libs/nccl/lib \
NCCL_LIBRARY=/share/modules/hpc_sdk/21.3/Linux_x86_64/21.3/comm_libs/nccl/lib/libnccl.so \
VERBOSE=1 \
V=1 \
nice python3 setup.py bdist_wheel |& tee sfantao_install.log

pip install dist/*.whl

#
# APEX (f9305e7)
#

cd $base
git clone https://github.com/NVIDIA/apex
cd apex 

CPATH=/share/modules/hpc_sdk/21.3/Linux_x86_64/21.3/cuda/11.2/targets/x86_64-linux/include:$CPATH \
LIBRARY_PATH=/share/modules/hpc_sdk/21.3/Linux_x86_64/21.3/cuda/11.2/targets/x86_64-linux/lib:$LIBRARY_PATH \
CC=gcc \
CXX=g++ \
pip wheel -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
pip install *.whl

#
# Create Megatron env
#
cd $base
conda create -n megatron --clone megatron-base
conda activate megatron
pip install pytorch/dist/*.whl
pip install apex/*.whl
conda install nltk pybind11

# 
# Checkout Megatron (e156d2f)
#
cd $base
git clone https://github.com/NVIDIA/Megatron-LM megatron
cd megatron
#
```

## Getting some data.
Sam tested the instalation with the RACE data set but also experimented with extracting wiki page info.
Sam used /tmp/sfantao to store the processed data has it was faster to access than the shared filesystem
he had available.

```
#
# Download data
#
mkdir data
cd data
curl -LO https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
curl -LO https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt
curl -LO https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt
curl -LO http://www.cs.cmu.edu/~glai1/data/race/RACE.tar.gz
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_uncased/zip -O megatron_bert_345m_v0.1_uncased.zip
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_cased/zip -O megatron_bert_345m_v0.1_cased.zip
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip -O megatron_lm_345m_v0.0.zip

#
# wiki preprocessing script
#
git clone https://github.com/attardi/wikiextractor.git -b v3.0.6
cd wikiextractor

rm -rf /tmp/sfantao
mkdir -p /tmp/sfantao/megatron-data/megatron_bert_345m_v0.1_uncased
cd /tmp/sfantao/megatron-data
tar -xf ~/lumi-builds/megatron/data/RACE.tar.gz 
cd /tmp/sfantao/megatron-data/megatron_bert_345m_v0.1_uncased
unzip ~/lumi-builds/megatron/data/megatron_bert_345m_v0.1_uncased.zip
cp ~/lumi-builds/megatron/data/enwiki-latest-pages-articles.xml.bz2 /tmp/sfantao/megatron-data

#
# Prepare wiki
#

mkdir -p /tmp/sfantao/megatron-data/wiki
cd /tmp/sfantao/megatron-data/wiki
cd /home/sfantao/lumi-builds/megatron/wikiextractor
python -m wikiextractor.WikiExtractor \
  /tmp/sfantao/megatron-data/enwiki-latest-pages-articles.xml.bz2 \
  -o /tmp/sfantao/megatron-data/wiki/preprocessed.json \
  --processes 64 \
  --json \
  -b 2G
```

## Running Megatron
 
Sam tried to run training on the RACE data with:
```
#
# Fix -V in /home/sfantao/lumi-builds/megatron/megatron/megatron/fused_kernels/__init__.py
#

CPATH=/share/modules/hpc_sdk/21.3/Linux_x86_64/21.3/cuda/11.2/targets/x86_64-linux/include:$CPATH \
LIBRARY_PATH=/share/modules/hpc_sdk/21.3/Linux_x86_64/21.3/cuda/11.2/targets/x86_64-linux/lib:$LIBRARY_PATH \
LD_LIBRARY_PATH=/share/modules/hpc_sdk/21.3/Linux_x86_64/21.3/cuda/11.2/targets/x86_64-linux/lib:$LD_LIBRARY_PATH \
MASTER_ADDR=localhost \
MASTER_PORT=12345 \
CC=gcc \
CXX=g++ \
../evaluate-bert-race.sh |& tee ../evaluate-bert-race.log
```

where `evaluate-bert-race.sh` is:

```
#!/bin/bash -e

TRAIN_DATA="/tmp/sfantao/megatron-data/RACE/train/middle"
VALID_DATA="/tmp/sfantao/megatron-data/RACE/dev/middle \
            /tmp/sfantao/megatron-data/RACE/dev/high"
VOCAB_FILE=/home/sfantao/lumi-builds/megatron/data/bert-large-uncased-vocab.txt 
PRETRAINED_CHECKPOINT=/tmp/sfantao/megatron-data/megatron_bert_345m_v0.1_uncased/release/mp_rank_00/model_optim_rng.pt
CHECKPOINT_PATH=/tmp/sfantao/megatron-data/megatron_bert_345m_v0.1_uncased/release/mp_rank_00/model_optim_rng_race.pt
COMMON_TASK_ARGS="--num-layers 24 \
                  --hidden-size 1024 \
                  --num-attention-heads 16 \
                  --seq-length 512 \
                  --max-position-embeddings 512 \
                  --fp16 \
                  --vocab-file $VOCAB_FILE"
                  
COMMON_TASK_ARGS_EXT="--train-data $TRAIN_DATA \
                      --valid-data $VALID_DATA \
                      --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
                      --activations-checkpoint-method uniform \
                      --save-interval 10000 \
                      --save $CHECKPOINT_PATH \
                      --log-interval 100 \
                      --eval-interval 1000 \
                      --eval-iters 10 \
                      --weight-decay 1.0e-1"


python tasks/main.py \
       --task RACE \
       $COMMON_TASK_ARGS \
       $COMMON_TASK_ARGS_EXT \
       --tokenizer-type BertWordPieceLowerCase \
       --epochs 3 \
       --micro-batch-size 4 \
       --lr 1.0e-5 \
       --lr-warmup-fraction 0.06
```

the end of the log showed:
```
-------------------------------------------------------------------------------------------------
 validation loss at iteration 19000 | lm loss value: 1.280714E+00 | lm loss PPL: 3.599210E+00 | 
-------------------------------------------------------------------------------------------------
saving checkpoint at iteration   19065 to /tmp/sfantao/megatron-data/megatron_bert_345m_v0.1_uncased/release/mp_rank_00/model_optim_rng_race.pt
  successfully saved checkpoint at iteration   19065 to /tmp/sfantao/megatron-data/megatron_bert_345m_v0.1_uncased/release/mp_rank_00/model_optim_rng_race.pt
calculating metrics ...
 > |epoch: 2| metrics for dev-middle: correct / total = 517 / 1436 = 36.0028 %, elapsed time (sec): 19.963
 > |epoch: 2| metrics for dev-high: correct / total = 1163 / 3451 = 33.7004 %, elapsed time (sec): 48.028
 >> |epoch: 2| overall: correct / total = 1680 / 4887 = 34.3769 %
done :-)
```
