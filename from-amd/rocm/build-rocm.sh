#!/bin/bash -ex

islockhart=false
if [ "$(uname -r)" = "5.3.18-150300.59.68_11.0.76-cray_shasta_c" ] ; then
  islockhart=true
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
  if $islockart ; then
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
  if $islockart ; then
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
  mkdir build
  cd build/
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
  if $islockart ; then
    sed -i "s#/opt/rocm/#/opt/rocm-5.0.2/#g" setup.py
  else
    sed -i "s#/opt/rocm/#/opt/rocm-5.2.0/#g" setup.py
  fi
  if $islockhart; then
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
  else
    echo "Download KB data."
    exit 1
  fi
}


prepare_data () {
  if [ -d /tmp/$(whoami)/megatron-data/megatron_bert_345m_v0.1_uncased ] ; then
    return
  fi
  
  mkdir -p /tmp/$(whoami)/megatron-data/megatron_bert_345m_v0.1_uncased
  cd /tmp/$(whoami)/megatron-data
  tar -xf $wd/data/RACE.tar.gz 
  cd /tmp/$(whoami)/megatron-data/megatron_bert_345m_v0.1_uncased
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
  
  MYSLURMID=$(squeue -u sfantao | tail -n 1 | awk '{print $1;}')
  export MYSLURMID
  srun --jobid=${MYSLURMID} -N 2 hostname
  
  cd $wd/megatron
  rm -rf megatron/fused_kernels/build/scaled_masked_softmax*
  
  #touch megatron/fused_kernels/scaled_softmax_cuda.cu
  #touch megatron/fused_kernels/scaled_masked_softmax_cuda.cu

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
   
  if $islockahrt ; then
    export LD_LIBRARY_PATH=$conda_location/envs/megatron-rocm-megatron-build/lib:$LD_LIBRARY_PATH
    
    srun --jobid=${MYSLURMID} -N 1 -n 1 --gpus=1 --cpu-bind=v,cores --cpus-per-task=16 \
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
  
  MYSLURMID=$(squeue -u $(whoami) | tail -n 1 | awk '{print $1;}')
  CHECKPOINT_PATH=checkpoints/bert_tiny
  DATA_PATH=$wd/data/kb/my-wordpiece_text_sentence
  VOCAB_FILE=$wd/data/kb/robin-vocab.txt
  
  export NNODES=1
  export NPROC_PER_NODE=1
  
  #!/bin/bash -x


<< EOF
export MASTER_ADDR=x1000c4s1b0n0
export MASTER_PORT=29500
export WORLD_SIZE=2
export OMP_NUM_THREADS=1

export NCCL_DEBUG=INFO 
export RCCL_KERNEL_COLL_TRACE_ENABLE=1 
export NCCL_DEBUG_SUBSYS=INIT,COLL
# export AMD_LOG_LEVEL=4 
export NCCL_PROTO=Simple
export NCCL_NET_GDR_LEVEL=0
export NCCL_MAX_NCHANNELS=2
export NCCL_SPINS_BEFORE_CHECK_ABORT=1000

export FI_LOG_LEVEL=info

#pkill python
#sleep 1

unset ROCR_VISIBLE_DEVICES
echo "ROCR_VISIBLE_DEVICES=\$ROCR_VISIBLE_DEVICES"
cmd="/tmp/$(whoami)/miniconda3/envs/mlperf-transformer-bf16/bin/python -u train.py --distributed-world-size=\$WORLD_SIZE --local_rank=\$SLURM_LOCALID /tmp/$(whoami)/data-transformer/transformer-mlperf-data/wmt14_en_de/utf8 --seed 11082 --arch transformer_wmt_en_de_big_t2t --bf16 --share-all-embeddings --optimizer adam --adam-betas (0.9,0.997) --adam-eps 1e-9 --clip-norm 0.0 --lr-scheduler inverse_sqrt --warmup-init-lr 0.0 --warmup-updates 1000 --lr 1.976e-3 --min-lr 0.0 --dropout 0.1 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 10240 --max-epoch 30 --target-bleu 25.0 --ignore-case --no-save --update-freq 1 --seq-len-multiple 2 --source_lang en --target_lang de --bucket_growth_factor 1.035 --batching_scheme v0p5_better --batch_multiple_strategy dynamic --max-len-a 1 --max-len-b 50 --lenpen 0.6 --no-progress-bar --dataloader-num-workers 2 --enable-dataloader-pin-memory --fast-xentropy --distributed-init-method env:// --distributed-weight-update 0 --dwu-num-blocks 4 --dwu-num-rs-pg 2 --dwu-num-ar-pg 2 --dwu-num-ag-pg 0 --dwu-overlap-reductions --dwu-num-chunks 1 --dwu-flat-mt --dwu-compute-L2-grad-norm --max-source-positions 64 --max-target-positions 64 --adam-betas (0.9,0.98)"
#cmd="/tmp/$(whoami)/miniconda3/envs/mlperf-transformer/bin/python -u train.py --distributed-world-size=\$WORLD_SIZE --local_rank=\$SLURM_LOCALID /tmp/$(whoami)/data-transformer/transformer-mlperf-data/wmt14_en_de/utf8 --seed 3475 --arch transformer_wmt_en_de_big_t2t --share-all-embeddings --optimizer adam --adam-betas (0.9,0.997) --adam-eps 1e-9 --clip-norm 0.0 --lr-scheduler inverse_sqrt --warmup-init-lr 0.0 --warmup-updates 1000 --lr 1.976e-3 --min-lr 0.0 --dropout 0.1 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 10240 --max-epoch 4 --target-bleu 25.0 --ignore-case --no-save --update-freq 1 --fp16 --seq-len-multiple 2 --source_lang en --target_lang de --bucket_growth_factor 1.035 --batching_scheme v0p5_better --batch_multiple_strategy dynamic --fast-xentropy --max-len-a 1 --max-len-b 50 --lenpen 0.6 --no-progress-bar --dataloader-num-workers 2 --enable-dataloader-pin-memory --multihead-attn-impl fast_with_lyrnrm_and_dropoutadd --distributed-init-method env:// --distributed-weight-update 0 --dwu-num-blocks 4 --dwu-num-rs-pg 2 --dwu-num-ar-pg 2 --dwu-num-ag-pg 0 --dwu-overlap-reductions --dwu-num-chunks 1 --dwu-flat-mt --dwu-compute-L2-grad-norm --max-source-positions 64 --max-target-positions 64 --adam-betas (0.9,0.98)"

export RANK=\$SLURM_PROCID
\$cmd |& tee ~/per-rank-\$RANK.log

EOF
  
  
  cat > helper.sh << EOF
#!/bin/bash -e
export MASTER_ADDR=\$(scontrol show hostname \$SLURM_NODELIST | head -n1)
export MASTER_PORT=34567
export OMP_NUM_THREADS=2
export WORLD_SIZE=\$(($NNODES*$NPROC_PER_NODE))
export RANK=\$SLURM_PROCID

#--local_rank=\$SLURM_LOCALID
unset ROCR_VISIBLE_DEVICES

DISTRIBUTED_ARGS="--nproc_per_node $NPROC_PER_NODE \
                  --nnodes \$SLURM_NNODES \
                  --node_rank \$SLURM_NODEID \
                  --master_addr \$MASTER_ADDR \
                  --master_port \$MASTER_PORT"

BERT_ARGS="--num-layers 12 \
           --hidden-size 768 \
           --num-attention-heads 12 \
           --seq-length 512 \
           --max-position-embeddings 512 \
           --lr 7e-4 \
           --train-iters 100000 \
           --lr-warmup-iters 1000 \
           --micro-batch-size 32 \
           --global-batch-size 2048 \
           --adam-beta2 0.999 \
           --adam-eps 1e-6 \
           --data-path $DATA_PATH \
           --vocab-file $VOCAB_FILE \
           --split 949,50,1 \
           --fp16 \
           --tokenizer-type BertWordPieceCase"

OUTPUT_ARGS="--log-interval 100 \
             --save-interval 5000 \
             --eval-interval 1000 \
             --eval-iters 10"

cmd="python3 -m torch.distributed.launch \$DISTRIBUTED_ARGS \
       $wd/megatron/pretrain_bert.py \
       \$BERT_ARGS \
       \$OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH"

echo "--> Rank \$SLURM_PROCID executing command: \$cmd"

\$cmd |& tee run-\$SLURM_PROCID.log
EOF
  chmod +x helper.sh 
  srun --jobid=$MYSLURMID -N $NNODES --gpus $((8*$NNODES)) ./helper.sh |& tee run-complete.log
}


#
# Run the various steps - uncomment all steps to build from scratch, 
# by default is configured to set environment and run only.
#

set_base_environment

checkout_rccl
checkout_libncurses
checkout_magma
checkout_pytorch
checkout_vision
checkout_apex
checkout_megatron
checkout_wikiextractor

build_rccl

conda_base
build_magma
set_magma_environment

conda_pytorch_build
build_pytorch
set_pytorch_environment

conda_vision_build
build_vision

conda_apex_build
build_apex
conda_megatron_build

download_data
download_kb_data
prepare_data
prepare_kb_data

#(run_training)
# (run_kb_pretraining)
