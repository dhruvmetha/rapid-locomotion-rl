#!/bin/bash
CONDA_DIR="$(conda info --base)"
source "${CONDA_DIR}/etc/profile.d/conda.sh"
# conda init bash
# conda activate base
conda activate "isaac_py38"
export LD_LIBRARY_PATH=/home/dhruv/miniconda3/envs/isaac_py38/lib
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
