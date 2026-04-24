#!/bin/bash
#SBATCH --job-name=train_head
#SBATCH --account=csci_ua_0480_042-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_head_%j.out
#SBATCH --error=logs/train_head_%j.err

NETID=am13367
BASE=/scratch/${NETID}
WORKDIR=${BASE}/assignment3
DATA_DIR=${WORKDIR}/cv_data_hw2
MATCH_DIR=${WORKDIR}/matched_annotations
OUT_DIR=${WORKDIR}/checkpoints/class_head_only

OVERLAY=${BASE}/overlay-25GB-500K.ext3
SIF=${BASE}/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif
VENV_DIR=${BASE}/venvs/assn3

mkdir -p ${OUT_DIR} ${WORKDIR}/logs

# FIX: :ro mode allows this to run alongside other jobs
singularity exec --bind /scratch --nv --overlay ${OVERLAY}:ro ${SIF} /bin/bash -lc "
set -euo pipefail
source ${VENV_DIR}/bin/activate

cd ${WORKDIR}
export PYTHONPATH=${WORKDIR}

python -u py_files/train.py \
  --base_dir ${DATA_DIR} \
  --matches_dir ${MATCH_DIR} \
  --output_dir ${OUT_DIR} \
  --batch_size 2 \
  --num_workers 4 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --epochs 10 \
  --val_split 0.2 \
  --freeze_mode class_head_only \
  --seed 42
"