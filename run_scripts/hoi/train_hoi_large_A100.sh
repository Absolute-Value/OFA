#!/usr/bin/env

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training). 
# Please set the options below according to the comments. 
# For multi-gpu workers training, these options should be manually set for each worker. 
# After setting the options, please run the script on each worker.
# To use the shuffled data (if exists), please uncomment the Line 24.

# Number of GPUs per GPU worker
GPUS_PER_NODE=4 
# Number of GPU workers, for single-worker training, please set to 1
WORKER_CNT=1
# The ip address of the rank-0 worker, for single-worker training, please set to localhost
export MASTER_ADDR=localhost
# The port for communication
export MASTER_PORT=8514
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
export RANK=0 

data_dir=/user/data/hico-det
data=${data_dir}/hico-det_train.tsv,${data_dir}/hico-det_val.tsv
restore_file=../../checkpoints/ofa_large.pt
selected_cols=0,1,2,3

log_dir=./hoi_logs/A100-80GBx4-01/
save_dir=./hoi_checkpoints/A100-80GBx4-01/
mkdir -p $log_dir $save_dir

bpe_dir=../../utils/BPE
user_dir=../../ofa_module

task=hoi_task
arch=ofa_large
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.1
batch_size=32
update_freq=1
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.2
decoder_drop_path_rate=0.2
dropout=0.1
attention_dropout=0.0
max_src_length=30
max_tgt_length=1000
num_bins=1000
max_hoi_num=24
echo "max_hoi_num "${max_hoi_num}

for max_epoch in 30 100; do
  echo "max_epoch "${max_epoch}
  for warmup_updates in {1000,}; do
    echo "warmup_updates "${warmup_updates} 
    for lr in {5e-5,}; do
      echo "lr "${lr}
      for patch_image_size in {512,}; do
        echo "patch_image_size "${patch_image_size}

        log_file=${log_dir}/${max_epoch}"_"${warmup_updates}"_"${lr}"_"${patch_image_size}"_rank"${RANK}".log"
        save_path=${save_dir}/${max_epoch}"_"${warmup_updates}"_"${lr}"_"${patch_image_size}
        mkdir -p $save_path

        python -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --nnodes=${WORKER_CNT} --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} ../../train.py \
          ${data} \
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
          --weight-decay=0.01 \
          --optimizer=adam \
          --adam-betas="(0.9,0.999)" \
          --adam-eps=1e-08 \
          --clip-norm=1.0 \
          --lr-scheduler=polynomial_decay \
          --lr=${lr} \
          --max-epoch=${max_epoch} \
          --warmup-updates=${warmup_updates} \
          --log-format=simple \
          --log-interval=40 \
          --fixed-validation-seed=7 \
          --no-epoch-checkpoints --keep-best-checkpoints=1 \
          --save-interval=5 --validate-interval=1 \
          --max-src-length=${max_src_length} \
          --max-tgt-length=${max_tgt_length} \
          --find-unused-parameters \
          --freeze-encoder-embedding \
          --freeze-decoder-embedding \
          --add-type-embedding \
          --scale-attn \
          --scale-fc \
          --scale-heads \
          --disable-entangle \
          --num-bins=${num_bins} \
          --patch-image-size=${patch_image_size} \
          --fp16 \
          --fp16-scale-window=512 \
          --max-hoi-num=${max_hoi_num} \
          --num-workers=0> ${log_file} 2>&1
      done
    done
  done
done