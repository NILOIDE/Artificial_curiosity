#!/bin/bash
conda create --name AC2 python=3.8 cudatoolkit=10.1 -y
eval "$(conda shell.bash hook)"
source activate AC2
pip install --user -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py

cd code/grid-gym
pip install -e .
cd ../..