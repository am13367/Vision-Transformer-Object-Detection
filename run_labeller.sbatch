#!/bin/bash
#SBATCH --job-name=labeller
#SBATCH --account=csci_ua_0480_042-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/labeller_%j.out
#SBATCH --error=logs/labeller_%j.err

# --- CONFIG FOR YOUR CURRENT FOLDER STRUCTURE ---
NETID=am13367
BASE_DIR=$(pwd)

# 1. Your python file is in the BASE folder, not src
SCRIPT_PATH=${BASE_DIR}/data_ground_truth_labeller.py

# 2. Your data is in the BASE folder
DATA_DIR=${BASE_DIR}/cv_data_hw2

# 3. Output
OUT_DIR=${BASE_DIR}/matched_annotations

# 4. Cache paths
HF_CACHE=/scratch/${NETID}/hf_cache
PIP_CACHE=/scratch/${NETID}/pip_cache
TMPDIR_S=/scratch/${NETID}/tmp
VENV_DIR=/scratch/${NETID}/venvs/assn3

mkdir -p ${BASE_DIR}/logs ${OUT_DIR} ${HF_CACHE} ${PIP_CACHE} ${TMPDIR_S}

OVERLAY=/scratch/${NETID}/overlay-25GB-500K.ext3
SIF=/scratch/${NETID}/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif

# --- EXECUTION ---
singularity exec --bind /scratch --nv --overlay ${OVERLAY}:rw ${SIF} /bin/bash -lc "
set -euo pipefail
export HF_HOME=${HF_CACHE}
export TRANSFORMERS_CACHE=${HF_CACHE}
export PIP_CACHE_DIR=${PIP_CACHE}
export TMPDIR=${TMPDIR_S}

source ${VENV_DIR}/bin/activate

echo \"Running Labeller...\"
echo \"Script: ${SCRIPT_PATH}\"
echo \"Data: ${DATA_DIR}\"

# IMPORTANT: This requires your Python file to have the argparse code we discussed!
python ${SCRIPT_PATH} --data_dir ${DATA_DIR} --output_dir ${OUT_DIR}

echo \"Done.\"
"