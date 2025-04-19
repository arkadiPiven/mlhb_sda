name=$0
. configs/controller.sh

args=" \
--zeroshot-frequency 1 \
--report-to tensorboard \
--train-data ../HPDv2/train/train.json $local_benchmark_path/annotations.json \
--val-data ../HPDv2/train/test.json $local_benchmark_path/annotations.json \
--train-folder ../HPDv2/train/train $local_benchmark_path \
--val-folder ../HPDv2/train/test $local_benchmark_path \
--warmup 500 \
--lr 0.0000033 \
--wd 0.35 \
--workers 4 4 \
--batch-size 16 16 \
--pretrained laion2B-s32B-b79K \
--dataset-type HPD ranking \
--ignore-in-train 0 1 \
--ignore-in-val 1 0 \
--train-data-sample-ratio 1.0 0 \
--model ViT-H-14 \
--lock-text \
--lock-image \
--lock-text-unlocked-layers 11 \
--lock-image-unlocked-groups 20 \
--logs none \
--light-augmentation \
--exp-name $name \
--iterations 100 \
"

eval "$header$args$extra_args 2>&1 | tee -a $work_dir/exp_$now.txt"
