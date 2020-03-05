#!/bin/bash
#SBATCH --time=2-23:59:59
#SBATCH --mem=20000
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nil.stoltanso@student.uva.nl
#SBATCH --output=name%j.out
export OMP_NUM_THREADS=1
module load matplotlib/2.1.2-foss-2018a-Python-3.6.4
module load TensorFlow/1.6.0-foss-2018a-Python-3.6.4
module load h5py/2.7.1-foss-2018a-Python-3.6.4
python test_ac_dqn.py