#!/bin/bash

#SBATCH -o /scratch/jl16973/few_shot_word_learning/slurm_outputs_gender_prompt/slurm-%j.out
#SBATCH -e /scratch/jl16973/few_shot_word_learning/slurm_outputs_gender_prompt/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=torch

module purge

singularity exec --nv \
	    --overlay /scratch/jl16973/few_shot_word_learning/my_pytorch.ext3 \
	    /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; python tmp2.py"
