#!/bin/bash -ex

base=$(pwd)
wd=$base
conda_location=$base/../miniconda3

#
# Modules to provide ROCm and recent GCC if needed
#
set_base_environment () {
  cd $wd
  ml nvhpc/21.3 gcc/9.3.0 numactl
  
  export CMAKE_PREFIX_PATH=$conda_location/envs/megatron-cuda-pytorch-build:/share/modules/hpc_sdk/21.3/Linux_x86_64/21.3/math_libs/11.2/targets/x86_64-linux

}

set_pytorch_environment () {
  export TORCH_USE_RTLD_GLOBAL=1
}

#
# Checkout components
#
checkout_cudnn () {
  rm -rf $wd/deps
  mkdir $wd/deps
  cd $wd/deps
  
  if [ -f $wd/cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive.tar.xz ] ; then
    tar -xf $wd/cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive.tar.xz
  else
    echo "You must download 'curl -LO https://developer.nvidia.com/compute/cudnn/secure/8.4.0/local_installers/11.6/cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive.tar.xz' after you authenticate to the NVIDIA website!"
    false
  fi 
}

checkout_pytorch () {
  cd $wd
  rm -rf pytorch
  git clone --recursive https://github.com/pytorch/pytorch
  cd pytorch
  git checkout -b v1.11.0 v1.11.0
  git submodule sync
  git submodule update --init --recursive --jobs 0
}
checkout_vision () {
  cd $wd
  rm -rf vision
  git clone https://github.com/pytorch/vision
  cd vision
  git checkout -b v0.12.0 v0.12.0
}
checkout_apex () {
  cd $wd
  rm -rf apex
  git clone https://github.com/NVIDIA/apex
  cd apex 
  git checkout -b mydev dcb02fcf805524b4df52e31d26953d852bbeb291
  git submodule sync
  git submodule update --init --recursive --jobs 0
}
checkout_megatron () {
  cd $wd
  rm -rf megatron
  git clone -o kb-private -b rocm https://github.com/kb-labb/Megatron-LuMi-private megatron
}
checkout_wikiextractor () { 
  cd $wd
  rm -rf wikiextractor
  git clone https://github.com/attardi/wikiextractor -b v3.0.6
}

#
# Set conda environments
#
conda_base () {
  cd $wd
  source $conda_location/bin/activate
  if [ ! -d $conda_location/envs/megatron-cuda-base ] ; then
    rm -rf tmp.yml
    echo "name: megatron-cuda-base" >> tmp.yml
    cat $wd/../conda.env.yml >> tmp.yml
    conda env create -f tmp.yml
    conda activate megatron-cuda-base
    conda install -y ninja pillow astunparse numpy pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
    conda install -y -c pytorch magma-cuda110 
  else
    conda activate megatron-cuda-base
  fi
}

conda_pytorch_build () {
  cd $wd
  if [ ! -d $conda_location/envs/megatron-cuda-pytorch-build ] ; then
    conda create -n megatron-cuda-pytorch-build --clone megatron-cuda-base
  fi
  conda activate megatron-cuda-pytorch-build
}

conda_vision_build () {
  cd $wd
  if [ ! -d $conda_location/envs/megatron-cuda-vision-build ] ; then
    conda create -n  megatron-cuda-vision-build --clone  megatron-cuda-pytorch-build
    conda activate  megatron-cuda-vision-build
    pip install $wd/pytorch/dist/*.whl
  else
    conda activate  megatron-cuda-vision-build
  fi
}

conda_apex_build () {
  cd $wd
  if [ ! -d $conda_location/envs/megatron-cuda-apex-build ] ; then
    conda create -n  megatron-cuda-apex-build --clone  megatron-cuda-vision-build
    conda activate  megatron-cuda-apex-build
    pip install $wd/vision/dist/*.whl
  else
    conda activate  megatron-cuda-apex-build
  fi
}

conda_megatron_build () {
  cd $wd
  if [ ! -d $conda_location/envs/megatron-cuda-megatron-build ] ; then
    conda create -n  megatron-cuda-megatron-build --clone  megatron-cuda-apex-build
    conda activate  megatron-cuda-megatron-build
    pip install $wd/apex/dist/*.whl
    conda install -y nltk pybind11
  else
    conda activate  megatron-cuda-megatron-build
  fi
}


#
# Build components 
#

build_pytorch () {
  cd $wd/pytorch
  nice python3 setup.py clean

  MKLROOT=$conda_location/envs/megatron-cuda-pytorch-build \
  CC=gcc \
  CXX=g++ \
  USE_CUDA=1 \
  USE_SYSTEM_NCCL=1 \
  USE_NCCL=1 \
  USE_CUDNN=1 \
  BLAS=MKL \
  TORCH_CUDA_ARCH_LIST="8.0" \
  CUDA_HOME=$(realpath $(dirname $(which ptxas))/../) \
  CUDNN_LIB_DIR=$wd/deps/cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive/lib \
  CUDNN_INCLUDE_DIR=$wd/deps/cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive/include \
  NCCL_INCLUDE_DIR=/share/modules/hpc_sdk/21.3/Linux_x86_64/21.3/comm_libs/nccl/include \
  NCCL_LIB_DIR=/share/modules/hpc_sdk/21.3/Linux_x86_64/21.3/comm_libs/nccl/lib \
  NCCL_LIBRARY=/share/modules/hpc_sdk/21.3/Linux_x86_64/21.3/comm_libs/nccl/lib/libnccl.so \
  VERBOSE=1 \
  V=1 \
  nice python3 setup.py bdist_wheel |& tee $(whoami)_install.log
}

build_vision () {
  cd $wd/vision
  FORCE_CUDA=1 \
  CC=gcc \
  CXX=g++ \
  TORCH_CUDA_ARCH_LIST="8.0" \
  CUDA_HOME=$(realpath $(dirname $(which ptxas))/../) \
  nice python3 setup.py bdist_wheel |& tee $(whoami)_install.log
}


build_apex () {
  cd $wd/apex
  CPATH=/share/modules/hpc_sdk/21.3/Linux_x86_64/21.3/cuda/11.2/targets/x86_64-linux/include:$CPATH \
  LIBRARY_PATH=/share/modules/hpc_sdk/21.3/Linux_x86_64/21.3/cuda/11.2/targets/x86_64-linux/lib:$LIBRARY_PATH \
  CC=gcc \
  CXX=g++ \
  CUDA_HOME=$(realpath $(dirname $(which ptxas))/../) \
  nice python setup.py bdist_wheel --cpp_ext --cuda_ext |& tee $(whoami)_install.log
}

#
# Prepare data
#
download_data () {
  mkdir -p $wd/data
  cd $wd/data
  #curl -LO https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
  curl -LO https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt
  curl -LO https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt
  curl -LO http://www.cs.cmu.edu/~glai1/data/race/RACE.tar.gz
  wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_uncased/zip -O megatron_bert_345m_v0.1_uncased.zip
  wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_cased/zip -O megatron_bert_345m_v0.1_cased.zip
  wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip -O megatron_lm_345m_v0.0.zip
}

prepare_data () {
  mkdir -p /tmp/$(whoami)/megatron-data/megatron_bert_345m_v0.1_uncased
  cd /tmp/$(whoami)/megatron-data
  tar -xf $wd/data/RACE.tar.gz 
  cd /tmp/$(whoami)/megatron-data/megatron_bert_345m_v0.1_uncased
  unzip $wd/data/megatron_bert_345m_v0.1_uncased.zip
}

#
# Run training
#
run_training() {
  export CPATH=/share/modules/hpc_sdk/21.3/Linux_x86_64/21.3/cuda/11.2/targets/x86_64-linux/include:$CPATH
  export LIBRARY_PATH=/share/modules/hpc_sdk/21.3/Linux_x86_64/21.3/cuda/11.2/targets/x86_64-linux/lib:$LIBRARY_PATH
  export CC=gcc
  export CXX=g++
  export CUDA_HOME=$(realpath $(dirname $(which ptxas))/../)
  
  cd $wd/megatron
  #rm -rf megatron/fused_kernels/build

  export MASTER_ADDR=$(hostname)
  export MASTER_PORT=12345
  export OMP_NUM_THREADS=1
  
  TRAIN_DATA="/tmp/$(whoami)/megatron-data/RACE/train/middle"
  VALID_DATA="/tmp/$(whoami)/megatron-data/RACE/dev/middle \
              /tmp/$(whoami)/megatron-data/RACE/dev/high"
  VOCAB_FILE=$wd/data/bert-large-uncased-vocab.txt 
  PRETRAINED_CHECKPOINT=/tmp/$(whoami)/megatron-data/megatron_bert_345m_v0.1_uncased/release/mp_rank_00/model_optim_rng.pt
  CHECKPOINT_PATH=/tmp/$(whoami)/megatron-data/megatron_bert_345m_v0.1_uncased/release/mp_rank_00/model_optim_rng_race.pt
  
  # Always start at random
  rm -rf $CHECKPOINT_PATH
  
  COMMON_TASK_ARGS="--num-layers 24 \
                    --hidden-size 1024 \
                    --num-attention-heads 16 \
                    --seq-length 512 \
                    --max-position-embeddings 512 \
                    --fp16 \
                    --vocab-file $VOCAB_FILE"
                  
  # originally this had  --activations-checkpoint-method uniform 
  COMMON_TASK_ARGS_EXT="--train-data $TRAIN_DATA \
                        --valid-data $VALID_DATA \
                        --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
                        --checkpoint-activations \
                        --save-interval 10000 \
                        --save $CHECKPOINT_PATH \
                        --log-interval 100 \
                        --eval-interval 1000 \
                        --eval-iters 10 \
                        --weight-decay 1.0e-1"

  MYCMD="python tasks/main.py
         --task RACE \
         $COMMON_TASK_ARGS \
         $COMMON_TASK_ARGS_EXT \
         --tokenizer-type BertWordPieceLowerCase \
         --epochs 3 \
         --micro-batch-size 4 \
         --lr 1.0e-5 \
         --lr-warmup-fraction 0.06"
         
  # Binding has to be adjusted as needed.
   
  (pkill python && sleep 3) || true
   
  if [ -z "$DISTRIBUTED_RUN" ] ; then
    rm -rf $wd/evaluate-bert-race.log
    
#     numactl --physcpubind=16-31,144-159  --membind=0 \
#     /usr/local/bin/nsys profile \
#       -o $wd/nsys.out \
#       --stats=true \
#       --sample=none \
#       --trace=cuda,nvtx,mpi \
#       --capture-range=none \
#     $MYCMD --exit-interval 100 |& tee  $wd/evaluate-bert-race.log
  
    numactl --physcpubind=16-31,144-159  --membind=0 \
    ncu \
      --target-processes all \
      --set full \
      --import-source yes \
      --kernel-regex-base function \
      --kernel-id ::regex::100 \
      -f --export all-kernels \
      $MYCMD --exit-interval 100 |& tee  $wd/evaluate-bert-race.log

#            --kernel-regex-base function \
#       --kernel-id ::regex:kCalcPMEOrthoNBFrc16_kernel:5000 -f \
#       --export kCalcPMEOrthoNBFrc16_kernel \
  else
     export WORLD_SIZE=2
     export TENSOR_MP_SIZE=2
     export PIPELINE_MP_SIZE=2
     MYCMD="$MYCMD \
            --tensor-model-parallel-size $TENSOR_MP_SIZE \
            --pipeline-model-parallel-size $PIPELINE_MP_SIZE \
            --sequence-parallel \
            --DDP-impl torch"

    rm -rf $wd/evaluate-bert-race-dist*
  
    pids=''
  
    RANK=0 \
    LOCAL_RANK=0 \
    numactl --physcpubind=16-31,144-159  --membind=0 \
    $MYCMD --local_rank 0 |& tee  $wd/evaluate-bert-race-dist-0.log &
    pids="$pids $!"
  
    RANK=1 \
    LOCAL_RANK=1 \
    numactl --physcpubind=0-15,128-143  --membind=1 \
    $MYCMD --local_rank 1 |& tee  $wd/evaluate-bert-race-dist-1.log &
    pids="$pids $!"
  
    for p in $pids ; do
      echo "Waiting for $p..."
      wait $p
      echo "Done!"
    done
  fi
}

#
# Run the various steps - uncomment all steps to build from scratch, 
# by default is configured to set environment and run only.
#

set_base_environment

# checkout_cudnn
# checkout_pytorch
# checkout_vision
# checkout_apex
# checkout_megatron
# checkout_wikiextractor

conda_base
conda_pytorch_build
# build_pytorch
set_pytorch_environment

conda_vision_build
# build_vision

conda_apex_build
# build_apex

conda_megatron_build

# download_data
# prepare_data

(run_training)
