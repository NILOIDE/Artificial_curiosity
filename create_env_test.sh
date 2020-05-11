#!/bin/bash
module purge
module load 2019
module load Anaconda3/2018.12

conda create --name AC python=3.8 cudatoolkit=10.1 -y
conda activate AC