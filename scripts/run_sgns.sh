#!/bin/bash
#SBATCH --array=70-71
#SBATCH -p rise # partition (queue)
#SBATCH --nodelist=pavia
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=6 # number of cores per task
#SBATCH --gres=gpu:2
#SBATCH -t 1-24:00 # time requested (D-HH:MM)

pwd
hostname
date
echo starting job...
source ~/.bashrc
conda activate lzhenv
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

root=/scratch/zhliu/repos/Word2Vec
cd ${root}

name=sgns
load=False
print_tofile=True
ckpt_path=/scratch/zhliu/checkpoints/${name}
datadir=${root}/data
window_size=2
unk='<UNK>'
max_vocab=100000
filename=text8.txt
e_dim=300
n_negs=20
epoch=100
batch_size=4096
ss_t=1e-5
cuda=True

mkdir -p ${ckpt_path}

cd src
pwd
CUDA_VISIBLE_DEVICES=6,7  python main.py \
    --name ${name} \
    --load ${load} \
    --print_tofile ${print_tofile} \
    --ckpt_path ${ckpt_path} \
    --datadir ${datadir} \
    --window_size ${window_size} \
    --unk ${unk} \
    --max_vocab ${max_vocab} \
    --filename ${filename} \
    --e_dim ${e_dim} \
    --n_negs ${n_negs} \
    --epoch ${epoch} \
    --batch_size ${batch_size} \
    --ss_t ${ss_t} \
    --cuda ${cuda} \
