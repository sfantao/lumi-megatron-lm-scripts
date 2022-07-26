# Stuff done on Vega


## Environment


**Loaded modules** when running `module load PyTorch`
```bash
module list

Currently Loaded Modules:
  1) GCCcore/10.3.0                    18) gompi/2021a                            35) SciPy-bundle/2021.05-foss-2021a            52) FriBidi/1.0.10-GCCcore-10.3.0
  2) zlib/1.2.11-GCCcore-10.3.0        19) FFTW/3.3.9-gompi-2021a                 36) typing-extensions/3.10.0.0-GCCcore-10.3.0  53) FFmpeg/4.3.2-GCCcore-10.3.0
  3) binutils/2.36.1-GCCcore-10.3.0    20) ScaLAPACK/2.1.0-gompi-2021a-fb         37) libyaml/0.2.5-GCCcore-10.3.0               54) libjpeg-turbo/2.0.6-GCCcore-10.3.0
  4) GCC/10.3.0                        21) foss/2021a                             38) PyYAML/5.4.1-GCCcore-10.3.0                55) jbigkit/2.1-GCCcore-10.3.0
  5) numactl/2.0.14-GCCcore-10.3.0     22) CUDA/11.3.1                            39) MPFR/4.1.0-GCCcore-10.3.0                  56) gzip/1.10-GCCcore-10.3.0
  6) XZ/5.2.5-GCCcore-10.3.0           23) Ninja/1.10.2-GCCcore-10.3.0            40) NASM/2.15.05-GCCcore-10.3.0                57) lz4/1.9.3-GCCcore-10.3.0
  7) libxml2/2.9.10-GCCcore-10.3.0     24) bzip2/1.0.8-GCCcore-10.3.0             41) x264/20210414-GCCcore-10.3.0               58) zstd/1.4.9-GCCcore-10.3.0
  8) libpciaccess/0.16-GCCcore-10.3.0  25) ncurses/6.2-GCCcore-10.3.0             42) LAME/3.100-GCCcore-10.3.0                  59) LibTIFF/4.2.0-GCCcore-10.3.0
  9) hwloc/2.4.1-GCCcore-10.3.0        26) libreadline/8.1-GCCcore-10.3.0         43) x265/3.5-GCCcore-10.3.0                    60) Pillow/8.2.0-GCCcore-10.3.0
 10) OpenSSL/1.1                       27) Tcl/8.6.11-GCCcore-10.3.0              44) expat/2.2.9-GCCcore-10.3.0                 61) cuDNN/8.2.1.32-CUDA-11.3.1
 11) libevent/2.1.12-GCCcore-10.3.0    28) SQLite/3.35.4-GCCcore-10.3.0           45) libpng/1.6.37-GCCcore-10.3.0               62) magma/2.6.1-foss-2021a-CUDA-11.3.1
 12) UCX/1.10.0-GCCcore-10.3.0         29) GMP/6.2.1-GCCcore-10.3.0               46) Brotli/1.0.9-GCCcore-10.3.0                63) GDRCopy/2.2-GCCcore-10.3.0
 13) libfabric/1.12.1-GCCcore-10.3.0   30) libffi/3.3-GCCcore-10.3.0              47) freetype/2.10.4-GCCcore-10.3.0             64) UCX-CUDA/1.10.0-GCCcore-10.3.0-CUDA-11.3.1
 14) PMIx/3.2.3-GCCcore-10.3.0         31) Python/3.9.5-GCCcore-10.3.0            48) util-linux/2.36-GCCcore-10.3.0             65) NCCL/2.10.3-GCCcore-10.3.0-CUDA-11.3.1
 15) OpenMPI/4.1.1-GCC-10.3.0          32) protobuf/3.17.3-GCCcore-10.3.0         49) fontconfig/2.13.93-GCCcore-10.3.0          66) expecttest/0.1.3-GCCcore-10.3.0
 16) OpenBLAS/0.3.15-GCC-10.3.0        33) protobuf-python/3.17.3-GCCcore-10.3.0  50) xorg-macros/1.19.3-GCCcore-10.3.0          67) PyTorch/1.10.0-foss-2021a-CUDA-11.3.1
 17) FlexiBLAS/3.0.4-GCC-10.3.0        34) pybind11/2.6.2-GCCcore-10.3.0          51) X11/20210518-GCCcore-10.3.0
```

**Further libraries** installed manually
```bash
pip install transformers tokenizers datasets nltk
```

**Tokenizers** have been trained with the [script](../scripts/train_tokenizer.py) for `wordpiece` and `bpe`.

Instead of extracting the vocab/merges we use the BigScience-Megatron-DeepSpeed changes to use HuggingFace _tokenizers_.

This covers three files:

- `megatron/tokenizer/tokenizer.py`
- `tools/preprocess_data.py`
- `megatron/data/indexed_dataset.py`

The data needs a **special `jsonl` format** which we get by running the [`doc_lines_to_json.py`](../scripts/doc_lines_to_json.py) script.

Finally we can run Megatron-LM's **preprocessing** script as follows:

```bash
python tools/preprocess_data.py \
    --input ~/group_space/data/text/public/wiki.sv.docs.filtered.lang.new.strict_095.dduped.json \
    --output-prefix my-wordpiece \
    --tokenizer-name-or-path ../LUMI-porting-Megatron-LM/tokenizers/wiki.sv.wordpiece.tokenizer \
    --dataset-impl mmap +
    --tokenizer-type PretrainedFromHF \
    --append-eod \
    --workers 8
```

## Megatron Pretraining with Singularity (BertWordPieceCase)

In order to pretrain with Megatron-LM we first need to:

1. Clone [Megatron-LuMi-private](https://github.com/kb-labb/Megatron-LuMi-private)
2. Download our small test dataset consisting of deduplicated Swedish Wikipedia: [wiki.sv.docs.filtered.lang.new.strict_095.dduped.json](https://kungliga-biblioteket.box.com/s/t2md4ryt4tejy6xexvyv13hyxabxk5ap). Click "Hämta" to download. 
3. Download vocabulary file [robin-vocab.txt](https://kungliga-biblioteket.box.com/s/2y0hmsnbuu4tknkt0tfazv5dkzkq95k6). Click "Hämta" to download. 
4. Build a singularity container from the definition file `megatron_new.def` in this repo folder. This will convert Nvidias NGC Pytorch container `nvcr.io/nvidia/pytorch:21.08-py3` to a Singularity container and install some additional packages (transformers, nltk, tokenizers, datasets). If you have sudo rightns on your system, you can build via `sudo singularity build megatron_new.sif megatron_new.def`. The resulting image will be named `megatron_new.sif`. If you do **not** have sudo privileges, you can ask an admin on HPC center for [fakeroot](https://docs.sylabs.io/guides/3.5/user-guide/fakeroot.html) privileges and build via the command `singularity build --fakeroot megatron_new.sif megatron_new.def`.
5. Preprocess our data file `wiki.sv.docs.filtered.lang.new.strict_095.dduped.json`. 

To preprocess the data file we use the preprocessing script `tools/preprocess_data.py`. An example bash script to launch the preprocessing script is provided in `preprocess_wordpiece.sh`. User needs to set correct path to the data file as `--input`, the vocab file path to `robin-vocab.txt` as `--vocab`. 

```bash
python tools/preprocess_data.py \
    --input ~/group/data/text/public/wiki.sv.docs.filtered.lang.new.strict_095.dduped.json \
    --output-prefix my-wordpiece \
    --vocab data/robin-vocab.txt \
    --dataset-impl mmap \
    --tokenizer-type BertWordPieceCase \
    --split-sentences \
    --workers 8
```

After the data preprocessing you should see two files named `my-wordpiece_text_sentence.idx` and `my-wordpiece_text_sentence.bin`. 

### Launch pretraining with Slurm

We should now be able to launch distributed training runs. Example launch scripts can be found in this repo under `/from-vega/distributed`.

On Vega we launch the job with

```bash
sbatch sbatch_run.sh
```

Adjust the following variables in `sbatch_run.sh` to point to your relevant working directory and the singularity container.

```
PROJECT=/ceph/hpc/home/eufatonr/group/faton/Megatron-LuMi
TARGET_DIR="/ceph/hpc/home/eufatonr/group/faton/Megatron-LuMi"
CONTAINER_PATH="/ceph/hpc/home/eufatonr/group/faton/Megatron-LuMi/megatron_new.sif"
LOGGING=$PROJECT/logs
```

Make sure a `logs` folder exists in the project directory before running. `mkdir logs`.

`sbatch_run.sh` will call the bash script `start_training.sh` which runs the script to launch distributed training. In `start_training.sh` the user should adjust paths to the vocab file and the prefix of the preprocessed data files. The case below assumes `sbatch_run.sh` was launched from the root of the project directory and that `my-wordpiece_text_sentence.idx` and `my-wordpiece_text_sentence.bin` are placed on the root level, as well as `robin-vocab.txt` being placed in a the `data/` folder under in the project directory. 

```
CHECKPOINT_PATH=checkpoints/bert_tiny
DATA_PATH=my-wordpiece_text_sentence
VOCAB_FILE=data/robin-vocab.txt
```

### TODO: Launch pretraining with Conda

Assuming we have a conda environment with Pytorch built with all necessary dependencies we should be able to launch a distributed training run in a similar manner as the Singularity contianer example above.

This is still work in progress. 

In `sbatch_run.sh` change

```
cmd="srun -l --output=$LOGGING/srun_$DATETIME.log \
      singularity exec --nv --pwd /ceph/hpc/home/eufatonr/group/faton/Megatron-LuMi --bind $PROJECT:$TARGET_DIR $CONTAINER_PATH \
      $TARGET_DIR/scripts/start_training.sh"
```

to 

```
cmd="srun -l --output=$LOGGING/srun_$DATETIME.log \
      $TARGET_DIR/scripts/start_training.sh"
```

### Logs

```
3:  iteration      100/  100000 | consumed samples:       204800 | elapsed time per iteration (ms): 1456.7 | learning rate: 5.670E-05 | global batch size:  2048 | lm loss: 9.483988E+00 | sop loss: 5.839754E-01 | loss scale: 16384.0 | grad norm: 2.940 | number of skipped iterations:  19 | number of nan iterations:   0 |
3: time (ms) | forward-compute: 993.38 | backward-compute: 393.38 | backward-params-all-reduce: 56.50 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 1.88 | optimizer-unscale-and-check-inf: 2.29 | optimizer-clip-main-grad: 1.83 | optimizer-copy-main-to-model-params: 1.39 | optimizer: 10.22 | batch-generator: 15.86
0: [Rank 0] (after 100 iterations) memory (MB) | allocated: 2793.5478515625 | max allocated: 19087.2451171875 | reserved: 24738.0 | max reserved: 24738.0
3:  iteration      200/  100000 | consumed samples:       409600 | elapsed time per iteration (ms): 819.8 | learning rate: 1.267E-04 | global batch size:  2048 | lm loss: 7.160607E+00 | sop loss: 1.887491E-01 | loss scale: 16384.0 | grad norm: 2.656 | number of skipped iterations:   0 | number of nan iterations:   0 |
3: time (ms) | forward-compute: 376.94 | backward-compute: 372.88 | backward-params-all-reduce: 56.78 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 1.87 | optimizer-unscale-and-check-inf: 1.23 | optimizer-clip-main-grad: 2.26 | optimizer-copy-main-to-model-params: 1.72 | optimizer: 10.53 | batch-generator: 5.94
3:  iteration      300/  100000 | consumed samples:       614400 | elapsed time per iteration (ms): 822.4 | learning rate: 1.967E-04 | global batch size:  2048 | lm loss: 6.222871E+00 | sop loss: 1.265377E-01 | loss scale: 16384.0 | grad norm: 2.969 | number of skipped iterations:   0 | number of nan iterations:   0 |
3: time (ms) | forward-compute: 379.57 | backward-compute: 373.26 | backward-params-all-reduce: 56.45 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 1.81 | optimizer-unscale-and-check-inf: 1.23 | optimizer-clip-main-grad: 2.25 | optimizer-copy-main-to-model-params: 1.71 | optimizer: 10.45 | batch-generator: 5.67
3:  iteration      400/  100000 | consumed samples:       819200 | elapsed time per iteration (ms): 820.9 | learning rate: 2.667E-04 | global batch size:  2048 | lm loss: 5.727469E+00 | sop loss: 1.089300E-01 | loss scale: 16384.0 | grad norm: 1.812 | number of skipped iterations:   0 | number of nan iterations:   0 |
3: time (ms) | forward-compute: 378.20 | backward-compute: 373.02 | backward-params-all-reduce: 56.62 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 1.81 | optimizer-unscale-and-check-inf: 1.23 | optimizer-clip-main-grad: 2.25 | optimizer-copy-main-to-model-params: 1.71 | optimizer: 10.45 | batch-generator: 6.55
3:  iteration      500/  100000 | consumed samples:      1024000 | elapsed time per iteration (ms): 818.1 | learning rate: 3.367E-04 | global batch size:  2048 | lm loss: 5.023194E+00 | sop loss: 1.022404E-01 | loss scale: 16384.0 | grad norm: 1.465 | number of skipped iterations:   0 | number of nan iterations:   0 |
3: time (ms) | forward-compute: 375.07 | backward-compute: 373.53 | backward-params-all-reduce: 56.36 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 1.84 | optimizer-unscale-and-check-inf: 1.24 | optimizer-clip-main-grad: 2.25 | optimizer-copy-main-to-model-params: 1.72 | optimizer: 10.50 | batch-generator: 5.93
3:  iteration      600/  100000 | consumed samples:      1228800 | elapsed time per iteration (ms): 823.3 | learning rate: 4.067E-04 | global batch size:  2048 | lm loss: 4.422433E+00 | sop loss: 9.685342E-02 | loss scale: 16384.0 | grad norm: 1.885 | number of skipped iterations:   0 | number of nan iterations:   0 |
3: time (ms) | forward-compute: 380.31 | backward-compute: 373.70 | backward-params-all-reduce: 56.22 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 1.84 | optimizer-unscale-and-check-inf: 1.24 | optimizer-clip-main-grad: 2.25 | optimizer-copy-main-to-model-params: 1.72 | optimizer: 10.48 | batch-generator: 5.95
3:  iteration      700/  100000 | consumed samples:      1433600 | elapsed time per iteration (ms): 822.5 | learning rate: 4.767E-04 | global batch size:  2048 | lm loss: 3.898950E+00 | sop loss: 8.721248E-02 | loss scale: 16384.0 | grad norm: 2.033 | number of skipped iterations:   0 | number of nan iterations:   0 |
3: time (ms) | forward-compute: 379.37 | backward-compute: 373.71 | backward-params-all-reduce: 55.97 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 2.02 | optimizer-unscale-and-check-inf: 1.26 | optimizer-clip-main-grad: 2.32 | optimizer-copy-main-to-model-params: 1.73 | optimizer: 10.80 | batch-generator: 6.15
3:  iteration      800/  100000 | consumed samples:      1638400 | elapsed time per iteration (ms): 822.5 | learning rate: 5.467E-04 | global batch size:  2048 | lm loss: 3.390602E+00 | sop loss: 8.353024E-02 | loss scale: 16384.0 | grad norm: 1.426 | number of skipped iterations:   0 | number of nan iterations:   0 |
3: time (ms) | forward-compute: 379.11 | backward-compute: 373.34 | backward-params-all-reduce: 56.72 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 1.93 | optimizer-unscale-and-check-inf: 1.25 | optimizer-clip-main-grad: 2.29 | optimizer-copy-main-to-model-params: 1.73 | optimizer: 10.65 | batch-generator: 5.70
3:  iteration      900/  100000 | consumed samples:      1843200 | elapsed time per iteration (ms): 819.2 | learning rate: 6.167E-04 | global batch size:  2048 | lm loss: 3.150133E+00 | sop loss: 7.888879E-02 | loss scale: 16384.0 | grad norm: 1.245 | number of skipped iterations:   0 | number of nan iterations:   0 |
3: time (ms) | forward-compute: 377.17 | backward-compute: 373.11 | backward-params-all-reduce: 55.68 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 1.87 | optimizer-unscale-and-check-inf: 1.24 | optimizer-clip-main-grad: 2.28 | optimizer-copy-main-to-model-params: 1.72 | optimizer: 10.56 | batch-generator: 5.53
3:  iteration     1000/  100000 | consumed samples:      2048000 | elapsed time per iteration (ms): 821.5 | learning rate: 6.867E-04 | global batch size:  2048 | lm loss: 2.984847E+00 | sop loss: 7.854642E-02 | loss scale: 16384.0 | grad norm: 1.110 | number of skipped iterations:   0 | number of nan iterations:   0 |
3: time (ms) | forward-compute: 378.78 | backward-compute: 373.63 | backward-params-all-reduce: 55.92 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 1.90 | optimizer-unscale-and-check-inf: 1.25 | optimizer-clip-main-grad: 2.22 | optimizer-copy-main-to-model-params: 1.72 | optimizer: 10.55 | batch-generator: 6.39
3: ------------------------------------------------------------------------------------------------------------------------------------------------------------
3:  validation loss at iteration 1000 | lm loss value: 4.328599E+00 | lm loss PPL: 7.583796E+01 | sop loss value: 1.302275E-01 | sop loss PPL: 1.139087E+00 | 
3: ------------------------------------------------------------------------------------------------------------------------------------------------------------
3:  iteration     1100/  100000 | consumed samples:      2252800 | elapsed time per iteration (ms): 908.0 | learning rate: 6.994E-04 | global batch size:  2048 | lm loss: 2.842843E+00 | sop loss: 7.651082E-02 | loss scale: 32768.0 | grad norm: 0.986 | number of skipped iterations:   0 | number of nan iterations:   0 |
3: time (ms) | forward-compute: 465.01 | backward-compute: 373.06 | backward-params-all-reduce: 56.40 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 1.88 | optimizer-unscale-and-check-inf: 1.33 | optimizer-clip-main-grad: 1.81 | optimizer-copy-main-to-model-params: 1.72 | optimizer: 10.19 | batch-generator: 10.18
3:  iteration     1200/  100000 | consumed samples:      2457600 | elapsed time per iteration (ms): 820.1 | learning rate: 6.987E-04 | global batch size:  2048 | lm loss: 2.743580E+00 | sop loss: 7.330114E-02 | loss scale: 32768.0 | grad norm: 0.740 | number of skipped iterations:   0 | number of nan iterations:   0 |
3: time (ms) | forward-compute: 377.60 | backward-compute: 373.69 | backward-params-all-reduce: 56.29 | backward-embedding-all-reduce: 0.02 | optimizer-copy-to-main-grad: 1.83 | optimizer-unscale-and-check-inf: 1.32 | optimizer-clip-main-grad: 1.47 | optimizer-copy-main-to-model-params: 1.72 | optimizer: 9.76 | batch-generator: 6.02
3:  iteration     1300/  100000 | consumed samples:      2662400 | elapsed time per iteration (ms): 822.6 | learning rate: 6.980E-04 | global batch size:  2048 | lm loss: 2.648705E+00 | sop loss: 6.989941E-02 | loss scale: 32768.0 | grad norm: 0.671 | number of skipped iterations:   0 | number of nan iterations:   0 |
3: time (ms) | forward-compute: 380.04 | backward-compute: 373.63 | backward-params-all-reduce: 56.30 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 1.93 | optimizer-unscale-and-check-inf: 1.34 | optimizer-clip-main-grad: 1.38 | optimizer-copy-main-to-model-params: 1.72 | optimizer: 9.81 | batch-generator: 5.70
3:  iteration     1400/  100000 | consumed samples:      2867200 | elapsed time per iteration (ms): 823.9 | learning rate: 6.973E-04 | global batch size:  2048 | lm loss: 2.578795E+00 | sop loss: 6.803597E-02 | loss scale: 32768.0 | grad norm: 0.626 | number of skipped iterations:   0 | number of nan iterations:   0 |
3: time (ms) | forward-compute: 381.16 | backward-compute: 373.79 | backward-params-all-reduce: 56.54 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 1.86 | optimizer-unscale-and-check-inf: 1.33 | optimizer-clip-main-grad: 1.34 | optimizer-copy-main-to-model-params: 1.72 | optimizer: 9.67 | batch-generator: 6.50
3:  iteration     1500/  100000 | consumed samples:      3072000 | elapsed time per iteration (ms): 819.5 | learning rate: 6.966E-04 | global batch size:  2048 | lm loss: 2.521821E+00 | sop loss: 6.540507E-02 | loss scale: 32768.0 | grad norm: 0.720 | number of skipped iterations:   0 | number of nan iterations:   0 |
3: time (ms) | forward-compute: 377.61 | backward-compute: 373.22 | backward-params-all-reduce: 56.15 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 1.88 | optimizer-unscale-and-check-inf: 1.34 | optimizer-clip-main-grad: 1.37 | optimizer-copy-main-to-model-params: 1.72 | optimizer: 9.75 | batch-generator: 6.64
```