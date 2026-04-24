#!/bin/bash
#SBATCH --job-name=eval_all
#SBATCH --account=csci_ua_0480_042-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_all_%j.out
#SBATCH --error=logs/eval_all_%j.err

NETID=am13367
BASE=/scratch/${NETID}
WORKDIR=${BASE}/assignment3
DATA_DIR=${WORKDIR}/cv_data_hw2
MATCH_DIR=${WORKDIR}/matched_annotations
OVERLAY=${BASE}/overlay-25GB-500K.ext3
SIF=${BASE}/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif
VENV_DIR=${BASE}/venvs/assn3

singularity exec --bind /scratch --nv --overlay ${OVERLAY}:ro ${SIF} /bin/bash -lc "
set -euo pipefail
source ${VENV_DIR}/bin/activate
cd ${WORKDIR}
export PYTHONPATH=${WORKDIR}

# ------------------------------------------------
# EVALUATE ALL FOUR MODELS
# ------------------------------------------------

echo '-------------------------------------------------'
echo 'EVALUATING: ALL PARAMS (BEST MODEL)'
python py_files/evaluation.py \
  --base_dir ${DATA_DIR} \
  --matches_dir ${MATCH_DIR} \
  --checkpoint checkpoints/all_params/best_model.pth \
  --score_thresh 0.1 \
  --iou_thresh 0.5

echo '-------------------------------------------------'
echo 'EVALUATING: CLASS HEAD ONLY'
python py_files/evaluation.py \
  --base_dir ${DATA_DIR} \
  --matches_dir ${MATCH_DIR} \
  --checkpoint checkpoints/class_head_only/best_model.pth \
  --score_thresh 0.1 \
  --iou_thresh 0.5

echo '-------------------------------------------------'
echo 'EVALUATING: TRANSFORMER ONLY'
python py_files/evaluation.py \
  --base_dir ${DATA_DIR} \
  --matches_dir ${MATCH_DIR} \
  --checkpoint checkpoints/transformer_only/best_model.pth \
  --score_thresh 0.1 \
  --iou_thresh 0.5

echo '-------------------------------------------------'
echo 'EVALUATING: BACKBONE ONLY'
python py_files/evaluation.py \
  --base_dir ${DATA_DIR} \
  --matches_dir ${MATCH_DIR} \
  --checkpoint checkpoints/backbone_only/best_model.pth \
  --score_thresh 0.1 \
  --iou_thresh 0.5
"