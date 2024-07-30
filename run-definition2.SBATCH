#!/bin/bash

#SBATCH -o /scratch/jl16973/few_shot_word_learning/slurm_outputs/slurm-%j.out
#SBATCH -e /scratch/jl16973/few_shot_word_learning/slurm_outputs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2  
#SBATCH --time=2:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=torch

module purge

singularity exec --nv \
	    --overlay /scratch/jl16973/few_shot_word_learning/my_env.ext3 \
	    /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; python definition_eval2.py"
