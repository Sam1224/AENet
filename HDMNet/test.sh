exp_name=$1     # split0/1/2/3
dataset=$2      # pascal/coco 
arch=$3         # HDMNetPlus
net=$4          # vgg/resnet50
postfix=$5      # manet/manet_5s

exp_dir=exp/${dataset}/${arch}/${exp_name}/${net} 
snapshot_dir=${exp_dir}/snapshot
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}_${net}_${postfix}.yaml
mkdir -p ${snapshot_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp test.sh test.py ${config} ${exp_dir}

echo ${arch}
echo ${config}

python test.py --config=${config} --arch=${arch}