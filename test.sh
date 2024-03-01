for dataset in deepfashion; do
    for model in tiny medium base large huge; do
        for size in 512; do # 256 384 480 512
            python test.py --dataset_name $dataset --img_size $size --model_size $model
        done
    done
done