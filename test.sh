for dataset in hico vcoco; do
    for model in tiny medium base large huge; do
        for size in 256 384 480 512; do
            python test.py --dataset_name $dataset --img_size $size --model_size $model
        done
    done
done