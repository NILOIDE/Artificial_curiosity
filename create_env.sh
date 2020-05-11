#!/bin/bash

module load 2019
module load Anaconda3/2018.12

conda create --name AC python=3.8 cudatoolkit=10.1 -y
. /sw/arch/Debian9/EB_production/2019/software/Anaconda3/2018.12/etc/profile.d/conda.sh
conda activate AC

pip install --user -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
# Install atari-py
pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py

# Install grid world envs
cd code/grid-gym
pip install -e .
cd ../..