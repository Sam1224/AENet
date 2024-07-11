exp_name=$1     # split0/1/2/3
dataset=$2      # pascal/coco
port=$3         # e.g., 1234
arch=$4         # SCCANPlus/CyCTRPlus
net=$5          # vgg/resnet50
postfix=$6      # ddp/ddp_5s

exp_dir=exp/${dataset}/${arch}/${exp_name}/${net}
snapshot_dir=${exp_dir}/snapshot
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}_${net}_${postfix}.yaml
mkdir -p ${snapshot_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp train.sh train.py ${config} ${exp_dir}

echo ${arch}
echo ${config}

python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=${port} train.py \
        --config=${config} \
        --arch=${arch} \
        2>&1 | tee ${result_dir}/train-$now.log
