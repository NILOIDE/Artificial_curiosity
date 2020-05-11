#!/bin/bash
module purge
module load 2019
module load Anaconda3/2018.12

conda create --name AC python=3.8 cudatoolkit=10.1 -y
. /sw/arch/Debian9/EB_production/2019/software/Anaconda3/2018.12/etc/profile.d/conda.sh
conda activate AC
conda list