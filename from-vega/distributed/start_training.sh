#!/bin/bash

pwd

export OMP_NUM_THREADS=2

CHECKPOINT_PATH=checkpoints/bert_tiny
DATA_PATH=my-wordpiece_text_sentence
VOCAB_FILE=data/robin-vocab.txt

DISTRIBUTED_ARGS="--nproc_per_node $NPROC_PER_NODE \
                  --nnodes $SLURM_JOB_NUM_NODES \
                  --node_rank $SLURM_NODEID \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

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


# cmd="python3 \
#        pretrain_bert.py \
#        $BERT_ARGS"

cmd="python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       $BERT_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH"


echo "Executing Command:"
echo $cmd

$cmd
