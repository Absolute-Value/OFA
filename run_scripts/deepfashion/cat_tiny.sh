#!/usr/bin/env

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training). 
# Please set the options below according to the comments. 
# For multi-gpu workers training, these options should be manually set for each worker. 
# After setting the options, please run the script on each worker.
# To use the shuffled data (if exists), please uncomment the Line 24.

# Number of GPUs per GPU worker
GPUS_PER_NODE=1
# Number of GPU workers, for single-worker training, please set to 1
WORKER_CNT=1
# The ip address of the rank-0 worker, for single-worker training, please set to localhost
export MASTER_ADDR=localhost
# The port for communication
export MASTER_PORT=8514
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
export RANK=0 

data_dir=/local/DeepFashion2
data=${data_dir}/train_ofa.tsv,${data_dir}/validation_ofa.tsv
restore_file=../../checkpoints/ofa_tiny.pt
selected_cols=0

log_dir=./logs/cat/tiny/
save_dir=./checkpoints/cat/tiny/
mkdir -p $log_dir $save_dir

bpe_dir=../../utils/BPE
user_dir=../../ofa_module

task=categorization
arch=ofa_tiny
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.1
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.2
decoder_drop_path_rate=0.2
dropout=0.1
attention_dropout=0.0
max_src_length=100
max_tgt_length=30
num_bins=1000
update_freq=(1 1 1 1)
batch_size=(192 96 48 32)
patch_image_size=(256 384 480 512)

for max_epoch in 30; do
  echo "max_epoch "${max_epoch}
  for warmup_updates in {1000,}; do
    for lr in {5e-5,}; do
      for ix in ${!batch_size[@]}; do
        echo "batch_size "${batch_size[ix]}
        echo "update_freq "${update_freq[ix]}
        echo "patch_image_size "${patch_image_size[ix]}

        log_file=${log_dir}/${max_epoch}"_"${warmup_updates}"_"${lr}"_"${patch_image_size[ix]}"_rank"${RANK}".log"
        save_path=${save_dir}/${max_epoch}"_"${warmup_updates}"_"${lr}"_"${patch_image_size[ix]}
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
          --batch-size=${batch_size[ix]} \
          --update-freq=${update_freq[ix]} \
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
          --patch-image-size=${patch_image_size[ix]} \
          --fp16 \
          --fp16-scale-window=512 \
          --num-workers=0> ${log_file} 2>&1
      done
    done
  done
done
