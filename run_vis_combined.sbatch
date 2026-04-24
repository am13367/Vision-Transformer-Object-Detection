#!/bin/bash
#SBATCH --job-name=vis_combined
#SBATCH --account=csci_ua_0480_042-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=logs/vis_combined_%j.out
#SBATCH --error=logs/vis_combined_%j.err

NETID=am13367
BASE=/scratch/${NETID}
WORKDIR=${BASE}/assignment3
DATA_DIR=${WORKDIR}/cv_data_hw2
MATCH_DIR=${WORKDIR}/matched_annotations

# Define output parent folder
VIS_ROOT=${WORKDIR}/vis_predictions

OVERLAY=${BASE}/overlay-25GB-500K.ext3
SIF=${BASE}/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif
VENV_DIR=${BASE}/venvs/assn3

# Create all subdirectories (mkdir -p is safe, it won't complain if they exist)
mkdir -p ${VIS_ROOT}/all_params
mkdir -p ${VIS_ROOT}/backbone_only
mkdir -p ${VIS_ROOT}/class_head_only
mkdir -p ${VIS_ROOT}/transformer_only
mkdir -p ${WORKDIR}/logs

singularity exec --bind /scratch --nv --overlay ${OVERLAY}:ro ${SIF} /bin/bash -lc "
set -euo pipefail
source ${VENV_DIR}/bin/activate
cd ${WORKDIR}
export PYTHONPATH=${WORKDIR}

# ---------------------------------------------------------
# FIX: Clean old images so we don't get duplicates/appending
# ---------------------------------------------------------
echo 'Cleaning old visualizations...'
rm -rf ${VIS_ROOT}/*

echo '--- Visualizing ALL PARAMS ---'
python py_files/visualize_preds.py \
  --base_dir ${DATA_DIR} \
  --matches_dir ${MATCH_DIR} \
  --checkpoint ${WORKDIR}/checkpoints/all_params/best_model.pth \
  --out_dir ${VIS_ROOT}/all_params \
  --num_samples 10 \
  --score_thresh 0.1

echo '--- Visualizing BACKBONE ONLY ---'
python py_files/visualize_preds.py \
  --base_dir ${DATA_DIR} \
  --matches_dir ${MATCH_DIR} \
  --checkpoint ${WORKDIR}/checkpoints/backbone_only/best_model.pth \
  --out_dir ${VIS_ROOT}/backbone_only \
  --num_samples 10 \
  --score_thresh 0.1

echo '--- Visualizing CLASS HEAD ONLY ---'
python py_files/visualize_preds.py \
  --base_dir ${DATA_DIR} \
  --matches_dir ${MATCH_DIR} \
  --checkpoint ${WORKDIR}/checkpoints/class_head_only/best_model.pth \
  --out_dir ${VIS_ROOT}/class_head_only \
  --num_samples 10 \
  --score_thresh 0.1

echo '--- Visualizing TRANSFORMER ONLY ---'
python py_files/visualize_preds.py \
  --base_dir ${DATA_DIR} \
  --matches_dir ${MATCH_DIR} \
  --checkpoint ${WORKDIR}/checkpoints/transformer_only/best_model.pth \
  --out_dir ${VIS_ROOT}/transformer_only \
  --num_samples 10 \
  --score_thresh 0.1

echo 'Done! All visualizations created.'
"