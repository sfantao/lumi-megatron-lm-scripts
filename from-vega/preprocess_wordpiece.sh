python tools/preprocess_data.py \
    --input ~/group/data/text/public/wiki.sv.docs.filtered.lang.new.strict_095.dduped.json \
    --output-prefix my-wordpiece \
    --vocab data/robin-vocab.txt \
    --dataset-impl mmap \
    --tokenizer-type BertWordPieceCase \
    --split-sentences \
    --workers 8