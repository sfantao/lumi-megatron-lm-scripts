# Piz Daint
This are the instructions to run the pretraining on [Piz Daint](https://www.cscs.ch/computers/piz-daint/). This is a Cray machine with one NVidia GPU per node.

Here we run the pretraining using singularity. We use the image [nvcr.io/nvidia/pytorch:21.07-py3](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) from Nvidia GPU Cloud. For more info, see the [release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_21-07.html#rel_21.07).

The container is based on OpenMPI-4.1.1rc1, but it shouldn't matter if all the communication is done via NCCL.

The image pulled with singularity has been named `pytorch_21.07-py3.sif` and it's on my `$SCRATCH` directory.

## The code
We save the kb-labb code somewhere on the host system. I do it on `$HOME/kblabb`
```bash
cd $HOME/kblabb
git clone git@github.com:kb-labb/Megatron-LuMi-private.git
git clone git@github.com:kb-labb/LUMI-porting-Megatron-LM.git
```

## Extra packages that are needed
Run the container interactively. Within the container go to a **host directory**. Create there a virtual environment and install the `transformers` package. For instance:
```bash
singularity exec -B $HOME:$HOME $SCRATCH/pytorch_21.07-py3.sif bash
cd $HOME/kblabb
python -m venv mlm-env --system-site-packages
. mlm-env/bin/activate
pip install transformers
```

## The data
* Download the following files as indicated on the main `README.md`:
  - the data: [`wiki.sv.docs.filtered.lang.new.strict_095.dduped.json`](https://kungliga-biblioteket.box.com/s/t2md4ryt4tejy6xexvyv13hyxabxk5ap)
  - the vocabulary: [`robin-vocab.txt`](https://kungliga-biblioteket.box.com/s/2y0hmsnbuu4tknkt0tfazv5dkzkq95k6)
* I put both files in a director named `data` inside my working director `$SCRATCH/pretrain-bert-mlm`

## Preprocess the data
```bash
singularity exec --nv \
          -B $SCRATCH:$SCRATCH \
          -B $HOME:$HOME \
          $SCRATCH/pytorch_21.07-py3.sif \
          bash -c ' \
          cd $SCRATCH/pretrain-bert-mlm/data;
          . $HOME/kblabb/mlm-env/bin/activate;
          python $HOME/kblabb/Megatron-LuMi-private/tools/preprocess_data.py \
          --input wiki.sv.docs.filtered.lang.new.strict_095.dduped.json \
          --output-prefix my-wordpiece \
          --tokenizer-type BertWordPieceCase \
          --vocab robin-vocab.txt \
          --dataset-impl mmap \
          --split-sentences \
          --workers 8'
```
That will create the files `my-wordpiece_text_sentence.bin` and `my-wordpiece_text_sentence.idx` inside the `data` directory.

## Run the pretraining
The training is run by the [start_training](https://github.com/kb-labb/LUMI-porting-Megatron-LM/blob/795f354b3a80fc34fa204f64b514dac55ccc6653/from-vega/distributed/start_training.sh) script. A couple of lines need to be changed to adapt it to Piz Daint:
```diff
diff --git a/from-vega/distributed/start_training.sh b/from-vega/distributed/start_training.sh
index 3c225e2..237d435 100644
--- a/from-vega/distributed/start_training.sh
+++ b/from-vega/distributed/start_training.sh
@@ -5,7 +5,7 @@ pwd
 export OMP_NUM_THREADS=2

 CHECKPOINT_PATH=checkpoints/bert_tiny
-DATA_PATH=my-wordpiece_text_sentence
+DATA_PATH=data/my-wordpiece_text_sentence
 VOCAB_FILE=data/robin-vocab.txt

 DISTRIBUTED_ARGS="--nproc_per_node $NPROC_PER_NODE \
@@ -22,8 +22,8 @@ BERT_ARGS="--num-layers 12 \
            --lr 7e-4 \
            --train-iters 100000 \
            --lr-warmup-iters 1000 \
-           --micro-batch-size 32 \
-           --global-batch-size 2048 \
+           --micro-batch-size 16 \
+           --global-batch-size 32 \
            --adam-beta2 0.999 \
            --adam-eps 1e-6 \
            --data-path $DATA_PATH \
@@ -43,7 +43,7 @@ OUTPUT_ARGS="--log-interval 100 \
 #        $BERT_ARGS"
 
  cmd="python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
-       pretrain_bert.py \
+       $HOME/kblabb/Megatron-LuMi-private/pretrain_bert.py \
        $BERT_ARGS \
        $OUTPUT_ARGS \
        --save $CHECKPOINT_PATH \
```
To run the pretraining, I got an interactive allocation `salloc ...` and run
```bash
export NPROC_PER_NODE=1
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=345687

srun singularity exec --nv \
          -B $SCRATCH:$SCRATCH \
          -B $HOME:$HOME \
          $SCRATCH/pytorch_21.07-py3.sif \
          bash -c ' \
          cd $SCRATCH/pretrain-bert-mlm;
          . $HOME/kblabb/mlm-env/bin/activate;
          export PYTHONPATH=$HOME/kblabb/Megatron-LuMi-private:$PYTHONPATH;
          bash $HOME/kblabb/LUMI-porting-Megatron-LM/from-vega/distributed/start_training.sh'
```

### 1 node
```
Executing Command:
python3 -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr nid02358 --master_port 53394 Megatron-LuMi-private/pretrain_bert.py --num-layers 12 --hidden-size 768 --num-attention-heads 12 --seq-length 512 --max-position-embeddings 512 --lr 7e-4 --train-iters 100000 --lr-warmup-iters 1000 --micro-batch-size 16 --global-batch-size 32 --adam-beta2 0.999 --adam-eps 1e-6 --data-path data/my-wordpiece_text_sentence --vocab-file data/robin-vocab.txt --split 949,50,1 --fp16 --tokenizer-type BertWordPieceCase --log-interval 100 --save-interval 5000 --eval-interval 1000 --eval-iters 10 --save checkpoints/bert_tiny --load checkpoints/bert_tiny

...
...
...

 iteration      100/  100000 | consumed samples:         3200 | elapsed time per iteration (ms): 2275.1 | learning rate: 5.880E-05 | global batch size:    32 | lm loss: 9.648865E+00 | sop loss: 6.989935E-01 | loss scale: 131072.0 | grad norm: 4.622 | number of skipped iterations:  16 | number of nan iterations:   0 |
[Rank 0] (after 100 iterations) memory (MB) | allocated: 2712.5478515625 | max allocated: 10595.82275390625 | reserved: 12906.0 | max reserved: 12906.0
time (ms) | forward-compute: 879.94 | backward-compute: 1373.40 | backward-params-all-reduce: 1.04 | backward-embedding-all-reduce: 0.02 | optimizer-copy-to-main-grad: 2.80 | optimizer-unscale-and-check-inf: 3.32 | optimizer-clip-main-grad: 3.36 | optimizer-copy-main-to-model-params: 2.39 | optimizer: 19.12 | batch-generator: 8.39
```

### 2 nodes
```
Executing Command:
python3 -m torch.distributed.launch --nproc_per_node 1 --nnodes 2 --node_rank 1 --master_addr nid02358 --master_port 53394 /users/sarafael/kblabb/Megatron-LuMi-private/pretrain_bert.py --num-layers 12 --hidden-size 768 --num-attention-heads 12 --seq-length 512 --max-position-embeddings 512 --lr 7e-4 --train-iters 100000 --lr-warmup-iters 1000 --micro-batch-size 16 --global-batch-size 32 --adam-beta2 0.999 --adam-eps 1e-6 --data-path data/my-wordpiece_text_sentence --vocab-file data/robin-vocab.txt --split 949,50,1 --fp16 --tokenizer-type BertWordPieceCase --log-interval 100 --save-interval 5000 --eval-interval 1000 --eval-iters 10 --save checkpoints/bert_tiny --load checkpoints/bert_tiny

...
...
...

[Rank 0] (after 100 iterations) memory (MB) | allocated: 2712.5478515625 | max allocated: 10595.8212890625 | reserved: 12906.0 | max reserved: 12906.0
 iteration      100/  100000 | consumed samples:         3200 | elapsed time per iteration (ms): 1308.8 | learning rate: 5.880E-05 | global batch size:    32 | lm loss: 9.647571E+00 | sop loss: 6.995057E-01 | loss scale: 131072.0 | grad norm: 4.585 | number of skipped iterations:  16 | number of nan iterations:   0 |
time (ms) | forward-compute: 465.50 | backward-compute: 689.03 | backward-params-all-reduce: 133.39 | backward-embedding-all-reduce: 0.04 | optimizer-copy-to-main-grad: 2.86 | optimizer-unscale-and-check-inf: 3.43 | optimizer-clip-main-grad: 3.38 | optimizer-copy-main-to-model-params: 2.40 | optimizer: 19.36 | batch-generator: 13.29

 iteration      200/  100000 | consumed samples:         6400 | elapsed time per iteration (ms): 1258.1 | learning rate: 1.267E-04 | global batch size:    32 | lm loss: 7.455150E+00 | sop loss: 3.953952E-01 | loss scale: 16384.0 | grad norm: 3.322 | number of skipped iterations:   3 | number of nan iterations:   0 |
time (ms) | forward-compute: 417.99 | backward-compute: 684.78 | backward-params-all-reduce: 133.27 | backward-embedding-all-reduce: 0.04 | optimizer-copy-to-main-grad: 2.87 | optimizer-unscale-and-check-inf: 2.82 | optimizer-clip-main-grad: 3.83 | optimizer-copy-main-to-model-params: 2.78 | optimizer: 20.59 | batch-generator: 1.01

 iteration      300/  100000 | consumed samples:         9600 | elapsed time per iteration (ms): 1278.1 | learning rate: 1.967E-04 | global batch size:    32 | lm loss: 6.941708E+00 | sop loss: 2.771287E-01 | loss scale: 16384.0 | grad norm: 2.434 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 437.11 | backward-compute: 684.97 | backward-params-all-reduce: 133.45 | backward-embedding-all-reduce: 0.04 | optimizer-copy-to-main-grad: 2.85 | optimizer-unscale-and-check-inf: 2.82 | optimizer-clip-main-grad: 3.96 | optimizer-copy-main-to-model-params: 2.87 | optimizer: 21.05 | batch-generator: 20.11
```

### 4 nodes (with global batch size 64)
```bash
Executing Command:
python3 -m torch.distributed.launch --nproc_per_node 1 --nnodes 4 --node_rank 3 --master_addr nid02275 --master_port 53394 /users/sarafael/kblabb/Megatron-LuMi-private/pretrain_bert.py --num-layers 12 --hidden-size 768 --num-attention-heads 12 --seq-length 512 --max-position-embeddings 512 --lr 7e-4 --train-iters 100000 --lr-warmup-iters 1000 --micro-batch-size 16 --global-batch-size 64 --adam-beta2 0.999 --adam-eps 1e-6 --data-path data/my-wordpiece_text_sentence --vocab-file data/robin-vocab.txt --split 949,50,1 --fp16 --tokenizer-type BertWordPieceCase --log-interval 100 --save-interval 5000 --eval-interval 1000 --eval-iters 10 --save checkpoints/bert_tiny --load checkpoints/bert_tiny

...
...
...

 iteration      100/  100000 | consumed samples:         6400 | elapsed time per iteration (ms): 1306.1 | learning rate: 5.740E-05 | global batch size:    64 | lm loss: 9.598613E+00 | sop loss: 6.551362E-01 | loss scale: 32768.0 | grad norm: 7.149 | number of skipped iterations:  18 | number of nan iterations:   0 |
[Rank 0] (after 100 iterations) memory (MB) | allocated: 2712.5478515625 | max allocated: 10595.82080078125 | reserved: 12906.0 | max reserved: 12906.0
time (ms) | forward-compute: 494.91 | backward-compute: 691.81 | backward-params-all-reduce: 98.71 | backward-embedding-all-reduce: 0.04 | optimizer-copy-to-main-grad: 2.87 | optimizer-unscale-and-check-inf: 3.42 | optimizer-clip-main-grad: 3.32 | optimizer-copy-main-to-model-params: 2.35 | optimizer: 19.13 | batch-generator: 6.84

 iteration      200/  100000 | consumed samples:        12800 | elapsed time per iteration (ms): 1228.9 | learning rate: 1.267E-04 | global batch size:    64 | lm loss: 7.458513E+00 | sop loss: 3.194392E-01 | loss scale: 16384.0 | grad norm: 10.413 | number of skipped iterations:   1 | number of nan iterations:   0 |
time (ms) | forward-compute: 419.65 | backward-compute: 687.83 | backward-params-all-reduce: 98.91 | backward-embedding-all-reduce: 0.04 | optimizer-copy-to-main-grad: 2.85 | optimizer-unscale-and-check-inf: 2.82 | optimizer-clip-main-grad: 3.91 | optimizer-copy-main-to-model-params: 2.84 | optimizer: 20.95 | batch-generator: 1.04

 iteration      300/  100000 | consumed samples:        19200 | elapsed time per iteration (ms): 1244.8 | learning rate: 1.967E-04 | global batch size:    64 | lm loss: 6.773415E+00 | sop loss: 2.008853E-01 | loss scale: 16384.0 | grad norm: 4.828 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 433.38 | backward-compute: 687.53 | backward-params-all-reduce: 101.26 | backward-embedding-all-reduce: 0.04 | optimizer-copy-to-main-grad: 2.85 | optimizer-unscale-and-check-inf: 2.81 | optimizer-clip-main-grad: 3.95 | optimizer-copy-main-to-model-params: 2.87 | optimizer: 21.09 | batch-generator: 1.01
```


A Slurm batch script could be
```bash
#!/bin/bash -l

#SBATCH --job-name=pretrain-bert
#SBATCH --time=00:20:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --account=<account>

module load daint-gpu
module load singularity

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export NPROC_PER_NODE=1
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=345687

srun singularity exec --nv \
          -B $SCRATCH:$SCRATCH \
          -B $HOME:$HOME \
          $SCRATCH/pytorch_21.07-py3.sif \
          bash -c ' \
          cd $SCRATCH/pretrain-bert-mlm;
          . $HOME/kblabb/mlm-env/bin/activate;
          export PYTHONPATH=$HOME/kblabb/Megatron-LuMi-private:$PYTHONPATH;
          bash $HOME/kblabb/LUMI-porting-Megatron-LM/from-vega/distributed/start_training.sh'
```

## Running native
The setup to run the pretraining native is pretty much the same. The only non-conventional thing is the installation of Nvidia's `apex`. After installing PyTorch (I installed pytorch-1.12.0 on venv):
```
. $SANDBOX/hpcpython2022/bin/activate
module load daint-gpu
module load cudatoolkit/21.5_11.3      # cuda-11.3 to match pytorch binaries
module switch PrgEnv-cray PrgEnv-gnu
module switch gcc gcc/9.3.0

git clone git@github.com:NVIDIA/apex.git

export TORCH_CUDA_ARCH_LIST=6.0
CXX=CC CC=cc pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
```
The posprocessing is
```bash
cd data
. $SANDBOX/hpcpython2022/bin/activate
python $HOME/kblabb/Megatron-LuMi-private/tools/preprocess_data.py \
       --input wiki.sv.docs.filtered.lang.new.strict_095.dduped.json \
       --output-prefix my-wordpiece \
       --tokenizer-type BertWordPieceCase \
       --vocab robin-vocab.txt \
       --dataset-impl mmap \
       --split-sentences \
       --workers 8
```
A Slurm batch script could be
```bash
#!/bin/bash -l

#SBATCH --job-name=pretrain-bert
#SBATCH --time=00:20:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --account=<account>

export NPROC_PER_NODE=1
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=14568

. $SANDBOX/hpcpython2022/bin/activate
module load daint-gpu
module load cudatoolkit/21.5_11.3                   # cuda-11.3 to match pytorch binaries
module switch PrgEnv-cray PrgEnv-gnu
module switch gcc gcc/9.3.0
export TORCH_CUDA_ARCH_LIST=6.0
export PYTHONPATH=$HOME/kblabb/native/Megatron-LuMi-private:$PYTHONPATH;

export CC=cc
export CXX=CC
srun bash $HOME/kblabb/native/LUMI-porting-Megatron-LM/from-vega/distributed/start_training.sh
```
I didn't see any difference in performancecompared to running with singularity. With the same training options as above (micro batch size 16 and global batch size 32), the elapsed time per iteration is ~ 2189.0 ms with one node and ~1186.3 ms for two nodes.
