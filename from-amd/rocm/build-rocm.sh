#!/bin/bash -ex

base=$(pwd)
wd=$base
conda_location=$base/../miniconda3

#
# Modules to provide ROCm and recent GCC if needed
#
set_base_environment () {
  cd $wd
  ml rocm/5.2.0-rel52 gcc/9.3.0 numactl

  #rl=/home/sfantao/lumi-builds/mlperf/rccl-plugin/rccl-install
  rl=$ROCM_PATH/rccl

  export ROCM_SOURCE_DIR="$ROCM_PATH"
  export RCCL_PATH="$rl"
  export HIPDIR="$HIP_PATH"
  export CMAKE_PREFIX_PATH="$rl/lib:$CMAKE_PREFIX_PATH"
  
  export LIBRARY_PATH=$LIBRARY_PATH:$wd/deps/usr/lib64
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$wd/deps/usr/lib64
}

set_magma_environment () {
  export MAGMA_HOME=$wd/magma-install
}

set_pytorch_environment () {
  export PYTORCH_ROCM_ARCH="gfx908;gfx90a"
  export CMAKE_PREFIX_PATH=$conda_location/envs/megatron-rocm-pytorch-build:$CMAKE_PREFIX_PATH
}

#
# Checkout components
#
checkout_libncurses () {
  rm -rf $wd/deps
  mkdir $wd/deps
  cd $wd/deps
  
  for i in \
    ncurses-devel-5.9-14.20130511.el7_4.x86_64.rpm \
    ncurses-libs-5.9-14.20130511.el7_4.x86_64.rpm \
  ; do
    curl -LO http://mirror.centos.org/centos/7/os/x86_64/Packages/$i
    rpm2cpio $i | cpio -idmv 
  done
}
checkout_magma () {
  cd $wd
  rm -rf magma magma-install
  git clone https://bitbucket.org/icl/magma
  cd magma
  git checkout -b mydev c62d700d880c7283b33fb1d615d62fc9c7f7ca21
}
checkout_pytorch () {
  cd $wd
  
  rm -rf pytorch
  git clone --recursive https://github.com/ROCmSoftwarePlatform/pytorch
  cd pytorch
  # based on rocm5.2_internal_testing
  # git checkout -b mydev 07b877b648a95bc4e2339c0cb17c59b629b9e7bd
  git checkout -b mydev 5d5e1e214555d824f29eebcd5eb72682359829d5
  git submodule sync
  git submodule update --init --recursive --jobs 0
}
checkout_vision () {
  cd $wd
  rm -rf vision
  git clone https://github.com/ROCmSoftwarePlatform/vision
  cd vision
  git checkout -b mydev e828eefa4c326f893ebdd07abae7adc873d6ab63
  cd -
}
checkout_apex () {
  cd $wd
  rm -rf apex
  git clone --recursive https://github.com/ROCmSoftwarePlatform/apex
  cd apex
  # git checkout -b mydev cf77e9b525e3a0f5b844387b73284df1a72c1ee6
  git checkout -b mydev 5de49cc90051adf094920675e1e21175de7bad1b
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
  if [ ! -d $conda_location/envs/megatron-rocm-base ] ; then
    rm -rf tmp.yml
    echo "name: megatron-rocm-base" >> tmp.yml
    cat $wd/../conda.env.yml >> tmp.yml
    conda env create -f tmp.yml
    conda activate megatron-rocm-base
    conda install -y ninja pillow
  else
    conda activate megatron-rocm-base
  fi
}

conda_pytorch_build () {
  cd $wd
  if [ ! -d $conda_location/envs/megatron-rocm-pytorch-build ] ; then
    conda create -n megatron-rocm-pytorch-build --clone megatron-rocm-base
  fi
  conda activate megatron-rocm-pytorch-build
}

conda_vision_build () {
  cd $wd
  if [ ! -d $conda_location/envs/megatron-rocm-vision-build ] ; then
    conda create -n megatron-rocm-vision-build --clone megatron-rocm-pytorch-build
    conda activate megatron-rocm-vision-build
    pip install $wd/pytorch/dist/*.whl
  else
    conda activate megatron-rocm-vision-build
  fi
}

conda_apex_build () {
  cd $wd
  if [ ! -d $conda_location/envs/megatron-rocm-apex-build ] ; then
    conda create -n megatron-rocm-apex-build --clone megatron-rocm-vision-build
    conda activate megatron-rocm-apex-build
    pip install $wd/vision/dist/*.whl
  else
    conda activate megatron-rocm-apex-build
  fi
}

conda_megatron_build () {
  cd $wd
  if [ ! -d $conda_location/envs/megatron-rocm-megatron-build ] ; then
    conda create -n megatron-rocm-megatron-build --clone megatron-rocm-apex-build
    conda activate megatron-rocm-megatron-build
    pip install $wd/apex/dist/*.whl
    conda install -y nltk pybind11
  else
    conda activate megatron-rocm-megatron-build
  fi
}


#
# Build components 
#
build_magma () {
  cd $wd/magma
  
  cp make.inc-examples/make.inc.hip-gcc-mkl make.inc
  echo 'LIBDIR += -L$(MKLROOT)/lib' >> make.inc
  echo "LIB += -Wl,--enable-new-dtags -Wl,--rpath,$ROCM_PATH/lib -Wl,--rpath,\$(MKLROOT)/lib -Wl,--rpath,$wd/magma-install/lib" >> make.inc
  echo 'DEVCCFLAGS += --gpu-max-threads-per-block=256' >> make.inc
  echo 'DEVCCFLAGS += --amdgpu-target=gfx90a --amdgpu-target=gfx908' >> make.inc
  sed -i 's/^FOPENMP/#FOPENMP/g' make.inc
  sed -i 's/VALID_GFXS = .*/VALID_GFXS = 908 90a/g' Makefile

  cd $wd/magma
  make -f make.gen.hipMAGMA -j 
  
  cd $wd/magma
  LANG="C.UTF-8" make lib/libmagma.so GPU_TARGET="gfx908 gfx90a" MKLROOT=$conda_location/envs/megatron-rocm-base -j

  cd $wd/magma
  make testing/testing_dgemm GPU_TARGET="gfx908 gfx90a"  MKLROOT=$conda_location/envs/megatron-rocm-base -j
  
  mv $wd/magma $wd/magma-install
}

build_pytorch () {
  cd $wd/pytorch
  
  sed -i 's#/opt/rocm/#/opt/rocm-5.2.0/#g' third_party/kineto/libkineto/CMakeLists.txt
  
  nice python3 setup.py clean
  
  nice python3 tools/amd_build/build_amd.py |& tee $(whoami)_amd_tunning.log

  CPATH=$CPATH:$ROCM_PATH/roctracer/include \
  CMAKE_PREFIX_PATH=$conda_location/envs/megatron-rocm-pytorch-build:$CMAKE_PREFIX_PATH \
  CMAKE_MODULE_PATH=$CMAKE_MODULE_PATH:$wd/pytorch/cmake/Modules_CUDA_fix \
  LDFLAGS='-ltinfo' \
  PYTORCH_ROCM_ARCH='gfx908;gfx90a' \
  RCCL_PATH=$rl \
  RCCL_DIR=$rl/lib/cmake/rccl \
  hip_DIR=${ROCM_PATH}/hip/cmake/ \
  VERBOSE=1 \
  V=1 \
  REL_WITH_DEB_INFO=1 \
  nice python3 setup.py bdist_wheel |& tee $(whoami)_install.log
}

build_vision () {
  cd $wd/vision
  FORCE_CUDA=1 nice python3 setup.py bdist_wheel |& tee $(whoami)_install.log
}

build_apex () {
  cd $wd/apex
  sed -i "s#/opt/rocm/#/opt/rocm-5.2.0/#g" setup.py
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
  # python -c "from torch.utils import cpp_extension ; print(cpp_extension.ROCM_HOME)"
  # 
  # return
  
  cd $wd/megatron
  #rm -rf megatron/fused_kernels/build

  #CPATH=/share/modules/hpc_sdk/21.3/Linux_x86_64/21.3/cuda/11.2/targets/x86_64-linux/include:$CPATH
  #LIBRARY_PATH=/share/modules/hpc_sdk/21.3/Linux_x86_64/21.3/cuda/11.2/targets/x86_64-linux/lib:$LIBRARY_PATH
  #LD_LIBRARY_PATH=/share/modules/hpc_sdk/21.3/Linux_x86_64/21.3/cuda/11.2/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
  export MASTER_ADDR=localhost
  export MASTER_PORT=12345
  #CC=gcc
  #CXX=g++
  
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

  # Binding has to be adjusted as needed.
  numactl --physcpubind=0-15,128-143  --membind=0 \
  python tasks/main.py \
         --task RACE \
         $COMMON_TASK_ARGS \
         $COMMON_TASK_ARGS_EXT \
         --tokenizer-type BertWordPieceLowerCase \
         --epochs 3 \
         --micro-batch-size 4 \
         --lr 1.0e-5 \
         --lr-warmup-fraction 0.06 |& tee  $wd/evaluate-bert-race.log
}

#
# Run the various steps - uncomment all steps to build from scratch, 
# by default is configured to set environment and run only.
#

set_base_environment

# checkout_libncurses
# checkout_magma
# checkout_pytorch
# checkout_vision
# checkout_apex
# checkout_megatron
# checkout_wikiextractor

conda_base
# build_magma
set_magma_environment

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
