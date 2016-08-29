#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=01:00:00
#SBATCH --mem=4096
#SBATCH --gres=gpu:K40:1

srun ./build/main data/trillion-half.csv 1 200 200
