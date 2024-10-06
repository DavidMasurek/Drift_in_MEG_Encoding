#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes 1

#SBATCH -p klab-cpu
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem 20G
#SBATCH --gres=gpu:0

#SBATCH --error=logs/slurm_logs/errors/error.o%j
#SBATCH --output=logs/slurm_logs/outputs/output.o%j
#SBATCH --job-name rsa-all-sess-combs
#SBATCH --mail-user=dmasurek@uos.de
#SBATCH --mail-type=ALL

echo "running in shell: " "$SHELL"
echo "*** loading spack modules ***"

export http{,s}_proxy=http://rhn-proxy.rz.uos.de:3128
spack load cuda@11.8.0
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate meg-encoding

srun python /share/klab/camme/camme/dmasurek/Drift_in_MEG_Encoding/src/main.py