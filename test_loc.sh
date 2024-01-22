for dataset in deepfashion; do
    for model in tiny medium large huge; do # tiny large huge
        for size in 256; do #  384 480 512
            python test_loc.py --dataset_name $dataset --img_size $size --model_size $model
        done
    done
done