#!/usr/bin/env

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=1051
export GPUS_PER_NODE=4

data_dir=/user/data/hico-det
data=${data_dir}/hico-det_train.tsv,${data_dir}/hico-det_test.tsv
restore_file=../../checkpoints/ofa_large.pt
selected_cols=0,1,2

log_dir=./hoi_logs/A6000x4-06/
save_dir=./hoi_checkpoints/A6000x4-06/
mkdir -p $log_dir $save_dir

bpe_dir=../../utils/BPE
user_dir=../../ofa_module

task=hoi_task
arch=ofa_large
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.1
lr=1e-5
batch_size=8
update_freq=1
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.1
decoder_drop_path_rate=0.1
dropout=0.1
attention_dropout=0.0
max_src_length=128
max_tgt_length=30
num_bins=1000
patch_image_size=480

for max_epoch in {10,}; do
  echo "max_epoch "${max_epoch}
  for warmup_ratio in {0.06,}; do
    echo "warmup_ratio "${warmup_ratio}
    for drop_worst_after in {2500,}; do
      echo "drop_worst_after "${drop_worst_after}

      log_file=${log_dir}/${max_epoch}"_"${warmup_ratio}"_"${drop_worst_after}".log"
      save_path=${save_dir}/${max_epoch}"_"${warmup_ratio}"_"${drop_worst_after}
      mkdir -p $save_path

      CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../train.py \
        ${data} \
        --selected-cols=${selected_cols} \
        --bpe-dir=${bpe_dir} \
        --user-dir=${user_dir} \
        --restore-file=${restore_file} \
        --reset-optimizer --reset-dataloader --reset-meters \
        --save-dir=${save_path} \
        --task=${task} \
        --arch=${arch} \
        --criterion=${criterion} \
        --label-smoothing=${label_smoothing} \
        --batch-size=${batch_size} \
        --update-freq=${update_freq} \
        --encoder-normalize-before \
        --decoder-normalize-before \
        --share-decoder-input-output-embed \
        --share-all-embeddings \
        --layernorm-embedding \
        --patch-layernorm-embedding \
        --code-layernorm-embedding \
        --resnet-drop-path-rate=${resnet_drop_path_rate} \
        --encoder-drop-path-rate=${encoder_drop_path_rate} \
        --decoder-drop-path-rate=${decoder_drop_path_rate} \
        --dropout=${dropout} \
        --attention-dropout=${attention_dropout} \
        --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=5.0 \
        --lr-scheduler=polynomial_decay --lr=${lr} \
        --max-epoch=${max_epoch} --warmup-ratio=${warmup_ratio} \
        --log-format=simple --log-interval=100 \
        --fixed-validation-seed=7 \
        --keep-last-epochs=15 \
        --save-interval=1 \
        --save-interval-updates=6000 \
        --disable-validation \
        --max-src-length=${max_src_length} \
        --max-tgt-length=${max_tgt_length} \
        --add-type-embedding \
        --scale-attn \
        --scale-fc \
        --scale-heads \
        --disable-entangle \
        --num-bins=${num_bins} \
        --patch-image-size=${patch_image_size} \
        --fp16 \
        --fp16-scale-window=512 \
        --num-workers=0> ${log_file} 2>&1
    done
  done
done