#!/bin/bash
export http{,s}_proxy=http://rhn-proxy.rz.uos.de:3128
spack load cuda@11.8.0
spack load miniconda3@4.10.3
eval "$(conda shell.bash hook)"
echo "Available conda environments are"
conda info --envs
default_env="meg-encoding"
read -p "Input your conda environment name (default: $default_env): " ENVNAME
ENVNAME=${ENVNAME:-$default_env}
echo "Activating conda environment: " $ENVNAME
conda activate $ENVNAME
clear