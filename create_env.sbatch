#!/bin/bash
#SBATCH --job-name=create_env
#SBATCH --account=csci_ua_0480_042-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/env_%j.out
#SBATCH --error=logs/env_%j.err

NETID=am13367
BASE=/scratch/${NETID}
VENV_DIR=${BASE}/venvs/assn3

OVERLAY=${BASE}/overlay-25GB-500K.ext3
SIF=${BASE}/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif

mkdir -p ${BASE}/logs $(dirname ${VENV_DIR})

singularity exec --bind /scratch --overlay ${OVERLAY}:rw ${SIF} /bin/bash -lc "
set -euo pipefail

# 1. Clean old env
rm -rf ${VENV_DIR}

# 2. Create venv WITHOUT pip (Fixes the ensurepip error)
python3 -m venv --without-pip --copies ${VENV_DIR}
source ${VENV_DIR}/bin/activate

# 3. Manually install pip
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
rm get-pip.py

# 4. Install dependencies
pip install --upgrade pip
pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
pip install --no-cache-dir numpy==1.26.4 scipy==1.12.0 scikit-learn==1.4.2 opencv-python-headless==4.9.0.80 transformers tqdm pillow timm

echo 'Environment setup complete at ${VENV_DIR}'
"