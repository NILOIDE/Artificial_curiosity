#!/bin/bash

conda env remove --n AC
conda create --name AC python=3.6 pytorch=1.2 torchvision cudatoolkit=10.1 -c pytorch -y
conda activate AC
pip install --user -r requirements.txt
pip uninstall atari-py -y
# Install atari-py
pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
cd code
python test_ac_dqn_2D.py
