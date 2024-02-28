#!/bin/bash
#
# Installer for PyTorch GPU environment with Pytorch+Cupy+PyTorch and CUDA 10.2
#
# Run: ./install_env.sh
#
# C. Birnie, 11/02/2024

echo 'Creating conda environment for self-supervised denoising tutorial'

# create conda env
conda env create -f environment_ssd.yml
conda activate ssd_tutorial
conda env list
echo 'Created and activated environment:' $(which python)

# check cupy works as expected
echo 'Checking pytorch is  correctly installed'
python -c 'import torch; mps_device = torch.device("mps"); print(torch.__version__);  print(torch.ones(10).to("mps"))'

echo 'Done!'