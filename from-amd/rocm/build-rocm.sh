#!/bin/bash -ex

islockhart=false
if [ "$(uname -r)" = "5.3.18-150300.59.68_11.0.76-cray_shasta_c" ] ; then
  islockhart=true
  echo "Building for Lockhart"
fi

islumi=false
if [[ "$(hostname)" == "nid"* ]] || [[ "$(hostname)" == "uan"* ]] ; then
  islumi=true
  islockhart=false
  echo "Building for LUMI"
fi

if $islockhart ; then
  mkdir -p /tmp/sfantao-megatron
  rm -f $(pwd)/tmp
  ln -s /tmp/sfantao-megatron tmp
  base=$(pwd)/tmp
  conda_location=$base/miniconda3
  cp -rf $(pwd)/../conda.env.yml $base
else
  base=$(pwd)
  conda_location=$base/../miniconda3
fi

wd=$base

#
# Modules to provide ROCm and recent GCC if needed
#
set_base_environment () {
  cd $wd
  
  if $islockhart ; then
    rl=$wd/rccl-install
        
    mkdir -p $wd/mymodules/myrocm
    cat > $wd/mymodules/myrocm/default.sh << EOF
#!/bin/bash -e

base="/opt/rocm-5.0.2"

subdirs=""
subdirs="\$subdirs hipblas hipcub hiprand hipsparse hsa miopen"
subdirs="\$subdirs oam opencl rocalution rocblas rocfft rocprim"
subdirs="\$subdirs rocrand rocsolver rocsparse rocthrust"

libdirs=""
libdirs="\$libdirs llvm lib64 lib llvm/lib hsa/lib hip/lib"


bindirs=""
bindirs="\$bindirs bin hip/bin atmi/bin"
bindirs="\$bindirs opencl/bin miopen/bin rocprofiler/bin llvm/bin"

for sname in \$subdirs ; do
  p="\$base/\$sname/include"
  export CMAKE_PREFIX_PATH="\$p:\$CMAKE_PREFIX_PATH"
  export CPATH="\$p:\$CPATH"
  export C_INCLUDE_PATH="\$p:\$C_INCLUDE_PATH"
  export CPLUS_INCLUDE_PATH="\$p:\$CPLUS_INCLUDE_PATH"
done

for dname in \$libdirs ; do
  p="\$base/\$dname"
  export LD_LIBRARY_PATH="\$p:\$LD_LIBRARY_PATH"
  export LD_RUN_PATH="\$p:\$LD_RUN_PATH"
  export LIBRARY_PATH="\$p:\$LIBRARY_PATH"
done

for dname in \$bindirs ; do
  p="\$base/\$dname"
  export PATH="\$p:\$PATH"
done 

export PKG_CONFIG_PATH="\$base/share/pkgconfig:\$PKG_CONFIG_PATH"
export MANPATH="\$base/share/man:\$MANPATH"
export ROCM_PATH="\$base"
export ROCM_SOURCE_DIR="\$base"
export RCCL_PATH="$rl"
export HIP_PATH="\$base/hip"
export HIPDIR="\$base/hip"

export CMAKE_PREFIX_PATH="$rl/lib:\$CMAKE_PREFIX_PATH"
export MAGMA_HOME=$wd/magma-install

export LIBRARY_PATH="$wd/deps/usr/lib64:\$LIBRARY_PATH"
export LD_LIBRARY_PATH="$wd/deps/usr/lib64:\$LD_LIBRARY_PATH"

export FI_CXI_ATS=0
export LD_LIBRARY_PATH="$rl/lib:$base/aws-ofi-rccl/src/.libs/:/opt/cray/libfabric/1.15.0.0/lib64/:\$LD_LIBRARY_PATH"

EOF

    export FI_CXI_ATS=0
    export LD_LIBRARY_PATH="/tmp/$(whoami)/rccl-plugin/rccl-install/lib:/tmp/$(whoami)/aws-ofi-rccl/src/.libs/:/opt/cray/libfabric/1.15.0.0/lib64/:/opt/rocm-$rocrel/lib:$LD_LIBRARY_PATH"
    source $wd/mymodules/myrocm/default.sh
    echo $ROCM_PATH
  elif $islumi ; then
    #export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/appl/lumi/SW/LUMI-21.12/common/EB/rocm/4.5.2/lib
    rl=$wd/rccl-install
    mkdir -p $wd/mymodules/myrocm
    cat > $wd/mymodules/myrocm/default.sh << EOF
#!/bin/bash -e

export PATH=$wd/valgrind/bin:$PATH

ml LUMI/22.06
ml partition/G
ml PrgEnv-gnu/8.3.3
# ml rocm/5.1.4

module unload cray-libsci

base="/appl/lumi/SW/LUMI-22.06/common/EB/rocm/5.1.4"
base="/pfs/lustrep2/projappl/project_462000125/samantao/rocm/rocm-5.2-65-sles"

subdirs=""
subdirs="\$subdirs hipblas hipcub hiprand hipsparse hsa miopen"
subdirs="\$subdirs oam opencl rocalution rocblas rocfft rocprim"
subdirs="\$subdirs rocrand rocsolver rocsparse rocthrust"

libdirs=""
libdirs="\$libdirs llvm lib64 lib llvm/lib hsa/lib hip/lib"


bindirs=""
bindirs="\$bindirs bin hip/bin atmi/bin"
bindirs="\$bindirs opencl/bin miopen/bin rocprofiler/bin llvm/bin"

for sname in \$subdirs ; do
  p="\$base/\$sname/include"
  export CMAKE_PREFIX_PATH="\$p:\$CMAKE_PREFIX_PATH"
  export CPATH="\$p:\$CPATH"
  export C_INCLUDE_PATH="\$p:\$C_INCLUDE_PATH"
  export CPLUS_INCLUDE_PATH="\$p:\$CPLUS_INCLUDE_PATH"
done

for dname in \$libdirs ; do
  p="\$base/\$dname"
  export LD_LIBRARY_PATH="\$p:\$LD_LIBRARY_PATH"
  export LD_RUN_PATH="\$p:\$LD_RUN_PATH"
  export LIBRARY_PATH="\$p:\$LIBRARY_PATH"
done

for dname in \$bindirs ; do
  p="\$base/\$dname"
  export PATH="\$p:\$PATH"
done 

export PKG_CONFIG_PATH="\$base/share/pkgconfig:\$PKG_CONFIG_PATH"
export MANPATH="\$base/share/man:\$MANPATH"
export ROCM_PATH="\$base"
export ROCM_SOURCE_DIR="\$base"
export RCCL_PATH="$rl"
export HIP_PATH="\$base/hip"
export HIPDIR="\$base/hip"

export CMAKE_PREFIX_PATH="$rl/lib:\$CMAKE_PREFIX_PATH"
export MAGMA_HOME=$wd/magma-install

export LIBRARY_PATH="$wd/deps/usr/lib64:$wd/deps/usr/lib64/ncurses5:\$LIBRARY_PATH"
export LD_LIBRARY_PATH="$wd/deps/usr/lib64:$wd/deps/usr/lib64/ncurses5:\$LD_LIBRARY_PATH"

export FI_CXI_ATS=0
export LD_LIBRARY_PATH="$rl/lib:$wd/aws-ofi-rccl/src/.libs/:/opt/cray/libfabric/1.15.0.0/lib64/:\$LD_LIBRARY_PATH"

EOF
    source $wd/mymodules/myrocm/default.sh
  else
    ml rocm/5.2.0-rel52 gcc/9.3.0 numactl
    #rl=/home/sfantao/lumi-builds/mlperf/rccl-plugin/rccl-install
    rl=$ROCM_PATH/rccl
  fi
  
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
  #export TORCH_USE_RTLD_GLOBAL=1
}

#
# Checkout components
#
checkout_rccl () {
  rm -rf $wd/aws-ofi-rccl  $wd/rccl  $wd/rccl-tests
  git clone -b cxi https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl 
  git clone https://github.com/ROCmSoftwarePlatform/rccl $wd/rccl
  git clone https://github.com/ROCmSoftwarePlatform/rccl-tests $wd/rccl-tests
}
checkout_libncurses () {
  if $islockhart ; then
    return
  fi
  
  if $islumi ; then
    rm -rf $wd/deps
    mkdir $wd/deps
    cd $wd/deps
  
   #       ncurses5-devel-6.1-5.6.2.x86_64.rpm \
   #   libncurses5-6.1-5.6.2.x86_64.rpm \
   # 
    for i in \
      libncurses6-6.1-5.6.2.x86_64.rpm \
      ncurses-devel-6.1-5.6.2.x86_64.rpm \
    ; do
      curl -LO https://download.opensuse.org/distribution/leap/15.3/repo/oss/x86_64/$i
      rpm2cpio $i | cpio -idmv 
    done
    return
  fi

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
  if $islockhart ; then
    tar -xf ~/rccl-repro/magma.tar.xz
    return
  fi
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
  
  cd third_party/ideep/mkl-dnn
  cat > mkl-dnn.patch << EOF
diff --git a/src/common/primitive_cache.cpp b/src/common/primitive_cache.cpp
index fb0cc7895..f5ee71420 100644
--- a/src/common/primitive_cache.cpp
+++ b/src/common/primitive_cache.cpp
@@ -152,7 +152,7 @@ void lru_primitive_cache_t::update_entry(
     //    by another thread
     // 2. After the requested entry had been evicted it was inserted again
     //    by another thread
-    if (it == cache_mapper_.end() || it->first.thread_id() != key.thread_id())
+    if (it == cache_mapper_.end() || !(it->first.thread_id() == key.thread_id()))
         return;
 
     const auto *op_desc = pd->op_desc();
EOF
  git apply < mkl-dnn.patch
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
  if [ ! -d $conda_location ] ; then
    curl -LO https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh
    bash Miniconda3-py39_4.10.3-Linux-x86_64.sh -b -p $conda_location -s
  fi
  source $conda_location/bin/activate
  if [ ! -d $conda_location/envs/megatron-rocm-base ] ; then
    rm -rf tmp.yml
    echo "name: megatron-rocm-base" >> tmp.yml
    cat $conda_location/../conda.env.yml >> tmp.yml
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
    conda install -y nltk pybind11 transformers==2.1.1
  else
    conda activate megatron-rocm-megatron-build
  fi
}


#
# Build components 
#
build_rccl () {

  cd $wd/rccl
  rm -rf build
  mkdir build
  cd build/
  rm -rf $wd/rccl-install 
  CXX=$ROCM_PATH/bin/hipcc cmake \
    -DCMAKE_INSTALL_PREFIX=$wd/rccl-install \
    -DAMDGPU_TARGETS="gfx90a:xnack-;gfx90a:xnack+" \
    -DCMAKE_BUILD_TYPE=Release \
    $wd/rccl
  nice make -j V=1 VERBOSE=1
  nice make -j install
  ln -s $wd/rccl-install/lib/librccl.so $wd/rccl-install/librccl.so

  cd $wd/rccl-tests
  sed -i 's/-std=c++14/-std=c++14 --amdgpu-target=gfx90a:xnack- --amdgpu-target=gfx90a:xnack+/g' $wd/rccl-tests/src/Makefile

  MPI_HOME=/opt/cray/pe/mpich/8.1.17/ofi/cray/10.0 ROCM_PATH=$ROCM_PATH \
    MPI=1 \
    NCCL_HOME=$wd/rccl-install \
    nice make -j

  cd $wd/aws-ofi-rccl
  ./autogen.sh
  echo "LD_LIBRARY_PATH->$LD_LIBRARY_PATH"
  CC=cc ./configure --with-libfabric=/opt/cray/libfabric/1.15.0.0 --enable-trace --with-hip=$ROCM_PATH --with-rccl=$wd/rccl-install
  nice make -j
}

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
  export PATH=~/tools/bin:$PATH

  sed -i "s#/opt/rocm/#$ROCM_PATH/#g" third_party/kineto/libkineto/CMakeLists.txt
  sed -i "s#/opt/rocm/#$ROCM_PATH/#g" cmake/Dependencies.cmake
  
  nice python3 setup.py clean
  
  nice python3 tools/amd_build/build_amd.py |& tee $(whoami)_amd_tunning.log
  CC=$(which cc) \
  CXX=$(which CC) \
  CPATH=$CPATH:$ROCM_PATH/roctracer/include \
  CMAKE_PREFIX_PATH=$conda_location/envs/megatron-rocm-pytorch-build:$CMAKE_PREFIX_PATH \
  CMAKE_MODULE_PATH=$CMAKE_MODULE_PATH:$wd/pytorch/cmake/Modules_CUDA_fix \
  LDFLAGS='-ltinfo' \
  PYTORCH_ROCM_ARCH='gfx90a' \
  RCCL_PATH=$rl \
  RCCL_DIR=$rl/lib/cmake/rccl \
  hip_DIR=${ROCM_PATH}/hip/cmake/ \
  VERBOSE=1 \
  V=1 \
  REL_WITH_DEB_INFO=1 \
  USE_MKLDNN=1 \
  nice python3 setup.py bdist_wheel |& tee $(whoami)_install.log
}

build_vision () {
  cd $wd/vision
  FORCE_CUDA=1 nice python3 setup.py bdist_wheel |& tee $(whoami)_install.log
}

build_apex () {
  cd $wd/apex
  if $islockart ; then
    sed -i "s#/opt/rocm/#/opt/rocm-5.0.2/#g" setup.py
  elif $islumi ; then
    sed -i "s#/opt/rocm/#$ROCM_PATH/#g" setup.py
  else
    sed -i "s#/opt/rocm/#/opt/rocm-5.2.0/#g" setup.py
  fi
  if $islockhart; then
    LD_LIBRARY_PATH=$conda_location/envs/megatron-rocm-megatron-build/lib:$LD_LIBRARY_PATH \
      nice python setup.py bdist_wheel --cpp_ext --cuda_ext |& tee $(whoami)_install.log
  elif $islumi; then
    LD_LIBRARY_PATH=$conda_location/envs/megatron-rocm-megatron-build/lib:$LD_LIBRARY_PATH \
    nice python setup.py bdist_wheel --cpp_ext --cuda_ext |& tee $(whoami)_install.log
  else
    nice python setup.py bdist_wheel --cpp_ext --cuda_ext |& tee $(whoami)_install.log
  fi
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

download_kb_data () {
  mkdir -p $wd/data/kb
  cd $wd/data/kb
  # Download wiki.sv.docs.filtered.lang.new.strict_095.dduped.json
  # from https://kungliga-biblioteket.box.com/s/t2md4ryt4tejy6xexvyv13hyxabxk5ap
  
  # Download robin-vocab.txt
  # from https://kungliga-biblioteket.box.com/s/2y0hmsnbuu4tknkt0tfazv5dkzkq95k6
  
  if $islockhart ; then
    cp /home/sfantao/lumi-builds/megatron/megatron-kb-lab-notes/kb-data/wiki.sv.docs.filtered.lang.new.strict_095.dduped.json .
    cp /home/sfantao/lumi-builds/megatron/megatron-kb-lab-notes/kb-data/robin-vocab.txt .
  elif $islumi ; then
    cp /pfs/lustrep4/projappl/project_462000075/samantao/ongoing/megatron/kb-data/wiki.sv.docs.filtered.lang.new.strict_095.dduped.json .
    cp /pfs/lustrep4/projappl/project_462000075/samantao/ongoing/megatron/kb-data/robin-vocab.txt .
  else
    echo "Download KB data."
    exit 1
  fi
}


prepare_data () {
  if [ -d  $wd/megatron-data/megatron_bert_345m_v0.1_uncased ] ; then
    return
  fi
  
  mkdir -p  $wd/megatron-data/megatron_bert_345m_v0.1_uncased
  cd  $wd/megatron-data
  tar -xf $wd/data/RACE.tar.gz 
  cd  $wd/megatron-data/megatron_bert_345m_v0.1_uncased
  unzip $wd/data/megatron_bert_345m_v0.1_uncased.zip
}

prepare_kb_data () {
  cd $wd/data/kb
  python $wd/megatron/tools/preprocess_data.py \
          --input wiki.sv.docs.filtered.lang.new.strict_095.dduped.json \
          --output-prefix my-wordpiece \
          --tokenizer-type BertWordPieceCase \
          --vocab robin-vocab.txt \
          --dataset-impl mmap \
          --split-sentences \
          --workers 8
  true
}

#
# Run training
#
run_training() {
  # python -c "from torch.utils import cpp_extension ; print(cpp_extension.ROCM_HOME)"
  # 
  # return
  
  MYSLURMID=$(squeue -u $(whoami) | tail -n 1 | awk '{print $1;}')
  export MYSLURMID
  srun --jobid=${MYSLURMID} -N 2 hostname
  
  cd $wd/megatron
  #rm -rf megatron/fused_kernels/build/scaled_masked_softmax*
  
  #touch megatron/fused_kernels/scaled_softmax_cuda.cu
  #touch megatron/fused_kernels/scaled_masked_softmax_cuda.cu

  export MASTER_ADDR=nid005107
  export MASTER_PORT=12345
  export OMP_NUM_THREADS=1
  
  TRAIN_DATA="$wd/megatron-data/RACE/train/middle"
  VALID_DATA="$wd/megatron-data/RACE/dev/middle \
              $wd/megatron-data/RACE/dev/high"
  VOCAB_FILE=$wd/data/bert-large-uncased-vocab.txt 
  PRETRAINED_CHECKPOINT=$wd/megatron-data/megatron_bert_345m_v0.1_uncased/release/mp_rank_00/model_optim_rng.pt
  CHECKPOINT_PATH=$wd/megatron-data/megatron_bert_345m_v0.1_uncased/release/mp_rank_00/model_optim_rng_race.pt
  
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
   
  if $islockhart; then
    export LD_LIBRARY_PATH=$conda_location/envs/megatron-rocm-megatron-build/lib:$LD_LIBRARY_PATH
    
    srun --jobid=${MYSLURMID} -N 1 -n 1 --gpus=1 --cpu-bind=v,cores --cpus-per-task=16 \
      $MYCMD --exit-interval 100 |& tee  $wd/evaluate-bert-race.log
  
    return
  fi 
  
  if $islumi; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$conda_location/envs/megatron-rocm-megatron-build/lib
    
    #MPICH_GPU_SUPPORT_ENABLED=1 \
    #srun --jobid=${MYSLURMID} -N 2 -n 2 --gpus=16 --cpus-per-task=8 \
    #/pfs/lustrep4/projappl/project_462000075/samantao/ongoing/HIP-Examples/gpu-burn/build/gpuburn-hip
    
    MPICH_GPU_SUPPORT_ENABLED=1 \
    srun --jobid=${MYSLURMID} -N 1 -n 1 --gpus=8 --cpus-per-task=8 \
    $MYCMD --exit-interval 100 |& tee  $wd/evaluate-bert-race.log
  
    return
  fi 
   
  if [ -z "$DISTRIBUTED_RUN" ] ; then
    rm -rf $wd/evaluate-bert-race.log
    
    rm -rf $wd/megatron.*
    numactl --physcpubind=0-15,128-143  --membind=0 \
    rocprof --stats --basenames on -i $wd/counters.txt -o $wd/megatron.csv \
    $MYCMD --exit-interval 100 |& tee  $wd/evaluate-bert-race.log
  
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
    numactl --physcpubind=0-15,128-143  --membind=0 \
    $MYCMD --local_rank 0 |& tee  $wd/evaluate-bert-race-dist-0.log &
    pids="$pids $!"
  
    RANK=1 \
    LOCAL_RANK=1 \
    numactl --physcpubind=16-31,144-159  --membind=1 \
    $MYCMD --local_rank 1 |& tee  $wd/evaluate-bert-race-dist-1.log &
    pids="$pids $!"
  
    for p in $pids ; do
      echo "Waiting for $p..."
      wait $p
      echo "Done!"
    done
  fi
  
}

run_kb_pretraining() {
  mkdir -p $wd/kb-runs
  cd $wd/kb-runs
  rm -rf run-*.log checkpoints
  
  MYSLURMID=$(squeue -u $(whoami) | tail -n 1 | awk '{print $1;}')
  CHECKPOINT_PATH=checkpoints/bert_tiny
  DATA_PATH=$wd/data/kb/my-wordpiece_text_sentence
  VOCAB_FILE=$wd/data/kb/robin-vocab.txt
  
  export NNODES=2
  export NPROC_PER_NODE=8
    
  cat > helper.sh << EOF
#!/bin/bash -ex

mpids=''
if [ \$SLURM_LOCALID -eq 0 ] ; then
  #rocm-monitor &
  mpids=\$!
fi

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
# 
# export NCCL_DEBUG=INFO 
# export RCCL_KERNEL_COLL_TRACE_ENABLE=1 
# export NCCL_DEBUG_SUBSYS=INIT,COLL
# # export AMD_LOG_LEVEL=4 
# export NCCL_PROTO=Simple
# export NCCL_NET_GDR_LEVEL=0
# export NCCL_MAX_NCHANNELS=2
# export NCCL_SPINS_BEFORE_CHECK_ABORT=1000
# export FI_LOG_LEVEL=info

export MASTER_ADDR=\$(scontrol show hostname "\$SLURM_NODELIST" | head -n1)
echo \$MASTER_ADDR

export MASTER_PORT=34567
export OMP_NUM_THREADS=2
export WORLD_SIZE=\$SLURM_NTASKS
export RANK=\$SLURM_PROCID
export LOCAL_RANK=\$SLURM_LOCALID

#--local_rank=\$SLURM_LOCALID
unset ROCR_VISIBLE_DEVICES

BERT_ARGS="--num-layers 12 \
           --hidden-size 768 \
           --num-attention-heads 12 \
           --seq-length 512 \
           --max-position-embeddings 512 \
           --lr 7e-4 \
           --train-iters 100000 \
           --lr-warmup-iters 1000 \
           --micro-batch-size 64 \
           --global-batch-size $((64*1*NPROC_PER_NODE*NNODES)) \
           --adam-beta2 0.999 \
           --adam-eps 1e-6 \
           --data-path $DATA_PATH \
           --vocab-file $VOCAB_FILE \
           --split 949,50,1 \
           --fp16 \
           --tokenizer-type BertWordPieceCase --local_rank \$LOCAL_RANK"

OUTPUT_ARGS="--log-interval 100 \
             --save-interval 5000 \
             --eval-interval 1000 \
             --eval-iters 10"
       
cmd="python3 \
       $wd/megatron/pretrain_bert.py \
       \$BERT_ARGS \
       \$OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH"

doprof=''
if [ \$SLURM_PROCID -eq 0 ] ; then
  rm -rf $wd/megatron.*
  #doprof="rocprof --stats --basenames on -i $wd/counters.txt -o $wd/megatron.csv"
  #doprof="rocprof --stats -i $wd/counters.txt -o $wd/megatron.csv"
  #doprof="rocprof --hip-trace -o $wd/megatron.csv"
  doprof="rocprof --stats -o $wd/megatron.csv"
fi

echo "--> Rank \$SLURM_PROCID (\$(taskset -p \$\$)) executing command: \$cmd"

\$doprof \$cmd --exit-interval 100 |& tee run-\$SLURM_PROCID.log

if [ ! "\$doprof" = '' ] ; then
  cd $wd
  rm -rf megatron.tar.xz
  tar -cf - megatron.* | xz -T32 > megatron.tar.xz
  cd -
fi

for p in \$mpids ; do
  kill \$p
done

EOF
  chmod +x helper.sh 
  #echo $LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=/users/samantao/coe/projappl/ongoing/megatron/megatron-kb-instructions/from-amd/rocm/rccl-install/lib:/opt/cray/libfabric/1.15.0.0/lib64/:/users/samantao/coe/projappl/ongoing/megatron/megatron-kb-instructions/from-amd/rocm/deps/usr/lib64:/users/samantao/coe/projappl/ongoing/megatron/megatron-kb-instructions/from-amd/rocm/deps/usr/lib64/ncurses5:/pfs/lustrep2/projappl/project_462000125/samantao/rocm/rocm-5.2-65-sles/hip/lib:/pfs/lustrep2/projappl/project_462000125/samantao/rocm/rocm-5.2-65-sles/hsa/lib:/pfs/lustrep2/projappl/project_462000125/samantao/rocm/rocm-5.2-65-sles/llvm/lib:/pfs/lustrep2/projappl/project_462000125/samantao/rocm/rocm-5.2-65-sles/lib:/pfs/lustrep2/projappl/project_462000125/samantao/rocm/rocm-5.2-65-sles/lib64:/pfs/lustrep2/projappl/project_462000125/samantao/rocm/rocm-5.2-65-sles/llvm:/opt/cray/pe/gcc/11.2.0/snos/lib64:/opt/cray/libfabric/1.15.0.0/lib64:/opt/cray/pe/papi/6.0.0.15/lib64:/users/samantao/coe/projappl/ongoing/megatron/megatron-kb-instructions/from-amd/rocm/deps/usr/lib64
  MASKS="ff000000000000,ff00000000000000,ff0000,ff000000,ff,ff00,ff00000000,ff0000000000"
  srun --jobid=$MYSLURMID -N $NNODES -n $((NNODES*NPROC_PER_NODE)) \
  --gpus=$((8*$NNODES)) \
  --cpus-per-task=8 --cpu-bind=mask_cpu:$MASKS \
  ./helper.sh |& tee run-complete.log
}

#
# Run the various steps - uncomment all steps to build from scratch, 
# by default is configured to set environment and run only.
#

set_base_environment

# checkout_rccl
# checkout_libncurses
# checkout_magma
# checkout_pytorch
# checkout_vision
# checkout_apex
# checkout_megatron
# checkout_wikiextractor
# 
# build_rccl

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
# download_kb_data
# prepare_data
# prepare_kb_data


MYSLURMID=$(squeue -u $(whoami) | tail -n 1 | awk '{print $1;}')
export MYSLURMID

#(run_training)
(run_kb_pretraining)
