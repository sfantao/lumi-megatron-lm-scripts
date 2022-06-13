# LUMI-porting-Megatron-LM

A repository for sharing documents, questions, milestones, plans, etc. that do not fit into the Megatron-LuMi fork.

Megatron-LuMi: https://github.com/kb-labb/Megatron-LuMi

## Goals

- get Megatron-LM running on
  - one GPU
  - one node
  - multiple nodes
- achieve throughputs similar to the reported ones
- figure out _good_ settings for
  - parallelism hyperparameters (e.g. tensor, pipeline, data/batch)
  - network hyperparameters (e.g. layer sizes, vocabulary size)
- get Megatron-LM variants running

## Data

- deduplicated
- cleaned with language identification using `fasttext`
  - \>95% for Swedish
- one document per line
- only allow text with "Swedish" characters (i.e. do not allow Cyrillic, Chinese, ...)

Preferably only use Swedish Wikipedia, but keep OSCAR as secondary choice.

### Swedish Wikipedia

- `1.2 GB` filesize
- `wc -lw`
  - 1,922,133 lines
  - 175,633,895 words

### Swedish OSCAR 2019

- `22 GB` filesize
- `wc -lw`
  - 9,040,248 lines
  - 3,605,324,835 words

## Megatron-LM Variants

### Megatron-Deepspeed

https://github.com/microsoft/Megatron-DeepSpeed

Microsoft's Megatron-LM extension using [DeepSpeed](https://www.deepspeed.ai/) using [ZeRO](https://arxiv.org/abs/1910.02054).
Their flagship model is [Megatron-Turing-NLG 530B](https://arxiv.org/abs/2201.11990).

### OPT

https://github.com/facebookresearch/metaseq

Meta's Megatron-LM extension using [FairScale](https://github.com/facebookresearch/fairscale), similar to DeepSpeed.
Meta has released multiple OPT models that are all [freely accesible](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT).

Especially interesting are their [chronicles](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT/chronicles) regarding the training process.

### BigScience

https://github.com/bigscience-workshop/Megatron-DeepSpeed

and

https://github.com/bigscience-workshop/bigscience

The BigScience code is an extension of the DeepSpeed extension.
Similar to the Meta repository we have again access to the chronicles that contain all the hidden knowledge.