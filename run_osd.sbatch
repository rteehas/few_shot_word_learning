#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=TeehanSNLI
#SBATCH --mail-type=END
#SBATCH --mail-user=rst306@nyu.edu
#SBATCH --output=slurm_%j.out

cd /scratch/rst306/few_shot_repo
singularity exec  \
	    --overlay /scratch/rst306/few_shot_word_learning/few_shot_model.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif\
	    /bin/bash -c "source /ext3/env.sh; python3 data/osd_data.py"
