#!/usr/bin/env bash
set -euo pipefail

# One-time HPC environment setup.
# Run this once before anything else on the cluster.

module purge
module load cuda/12.1.1
module load anaconda3/2024.06

ENV_NAME="lits-seg"
ENV_FILE="environment.yaml"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment ${ENV_NAME} already exists. Updating..."
    conda env update --name "${ENV_NAME}" --file "${ENV_FILE}" --prune
else
    echo "Creating environment ${ENV_NAME}..."
    conda env create --name "${ENV_NAME}" --file "${ENV_FILE}"
fi

conda activate "${ENV_NAME}"

echo "Verifying imports..."
python -c "import torch;                        print('torch:',   torch.__version__)"
python -c "import torch;                        print('CUDA:',    torch.cuda.is_available())"
python -c "import nibabel;                      print('nibabel:', nibabel.__version__)"
python -c "import wandb;                        print('wandb:',   wandb.__version__)"
python -c "import segmentation_models_pytorch;  print('smp:',     segmentation_models_pytorch.__version__)"
echo "Setup complete."
