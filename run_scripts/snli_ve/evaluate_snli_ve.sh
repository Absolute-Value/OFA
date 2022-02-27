#!/usr/bin/env bash

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

# dev or test
split=$1

data=../../dataset/snli_ve_data/snli_ve_${split}.tsv
path=../../checkpoints/snli_ve_large_best.pt
result_path=../../results/snli_ve
selected_cols=0,2,3,4,5

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=snli_ve \
    --batch-size=8 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"