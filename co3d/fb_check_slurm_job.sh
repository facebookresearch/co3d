#!/bin/bash
#SBATCH --mail-user=dnovotny@fb.com
#SBATCH --mail-type=all
#SBATCH --output=/checkpoint/%u/jobs/%x_%j.out
#SBATCH --error=/checkpoint/%u/jobs/%x_%j.err
#SBATCH --signal=USR1@30
#SBATCH --partition=learnfair
#SBATCH --time=4000
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=20
#SBATCH --mem="196G"
#SBATCH --job-name=co3d_check_job
source /etc/profile
module purge
module load anaconda3
pythonbin=/private/home/dnovotny/.conda/envs/co3d_env/bin/python
cd /private/home/dnovotny/co3d/
srun $pythonbin ./fb_check_script.py