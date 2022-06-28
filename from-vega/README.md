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