#!/bin/bash
#SBATCH --time=0-00:05:00
#SBATCH --mem=20000M
#SBATCH --nodes=1
#SBATCH --partition=gpu_short
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nil.stoltanso@student.uva.nl
#SBATCH --output=name%j_vae.out
module purge
module load 2019
module load Anaconda3/2018.12

# run conda and activate the thesis environment
. /sw/arch/Debian9/EB_production/2019/software/Anaconda3/2018.12/etc/profile.d/conda.sh
conda activate AC

export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATH
cd ..
srun python3 test_ac_dqn_2D.py --name=2Dcuriosity_LISA --z_dim=(512,) --wm_h_dim=(256,) --encoder_type='vae' --wm_opt='adam' --wm_lr=1e-4 --seed=1