#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=0-8:00
#SBATCH --output=out/%N-%j.out
#SBATCH --job-name=22vgg13-vgg8-0.0005-5-wttm
#SBATCH --account=def-ehyangit

module load StdEnv/2020
module load python/3.8
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
pip list 

export PYTHONUNBUFFERED=1

# model="wrn_40_2"
model="vgg13"
student_list=(
  # "wrn_40_1_aux"
  "ShuffleV1_aux"
  # wrn_40_1
  # ShuffleV1
)
echo "Starting script..."
JPEG_learning_rates=(0.0005)

JPEG_alphas=(20)

# alpha_it=1.01 # Same Arch
# alpha_it=1.50 # Different Arch

Temp=4
kl_coeff=0.9
ce_coeff=$(python -c "print(1.0 - $kl_coeff)")
GPU_ID=0

process_id=9

for student in "${student_list[@]}"; do
  for trial in 2; do
    for JPEG_alpha in "${JPEG_alphas[@]}"; do
      for JPEG_learning_rate in "${JPEG_learning_rates[@]}"; do
        model_path="./models/${model}_vanilla/ckpt_epoch_240.pth"
        CUDA_VISIBLE_DEVICES="${GPU_ID}" python3.8 train_student_jpeg.py \
            --tarch wrn_40_2_aux \
            --arch ${student} \
            --tcheckpoint ./checkpoint/train_teacher_cifar_arch_wrn_40_2_aux_dataset_cifar100_seed0/wrn_40_2_aux_22.pth.tar \
            --checkpoint-dir ./checkpoint \
            --data ./data \
            --gpu 0 \
            --manual 0 \
            --JPEG_learning_rate ${JPEG_learning_rate} \
            --JPEG_alpha ${JPEG_alpha} \
            --num_bit 11 --lr_decay_rate 0.1 --init-lr 0.01 --trial ${trial}
      done
    done
  done
done
