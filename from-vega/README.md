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