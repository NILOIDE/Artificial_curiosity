#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --mem=20000M
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nil.stoltanso@student.uva.nl
#SBATCH --output=name%j_cont.out
module purge
module load 2019
module load Anaconda3/2018.12

# run conda and activate the thesis environment
. /sw/arch/Debian9/EB_production/2019/software/Anaconda3/2018.12/etc/profile.d/conda.sh
conda activate AC

export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATH
cd ..
srun python3 test_ac_dqn_2D.py --env_name='Riverraid-v0' --z_dim='(512,)' --wm_h_dim='(256,)' --neg_samples=10 --hinge_value=0.1 --encoder_type='cont' --wm_opt='adam' --wm_lr=1e-4 --wm_target_net_steps=1000 --wm_enc_lr=1e-3 --seed=2 --train_steps=10000000