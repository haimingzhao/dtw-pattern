#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=12000
#SBATCH --gres=gpu:K40:1


srun ./build/main data/trillion-processed.csv 1 200 200 
srun ./build/main data/trillion-processed.csv 1 200 200 5500
srun ./build/main data/trillion-processed.csv 1 200 200 4500
srun ./build/main data/trillion-processed.csv 1 200 200 4000
srun ./build/main data/trillion-processed.csv 1 200 200 3500
srun ./build/main data/trillion-processed.csv 1 200 200 3250
srun ./build/main data/trillion-processed.csv 1 200 200 3000
srun ./build/main data/trillion-processed.csv 1 200 200 2750
srun ./build/main data/trillion-processed.csv 1 200 200 2500
srun ./build/main data/trillion-processed.csv 1 200 200 2250
srun ./build/main data/trillion-processed.csv 1 200 200 2000
srun ./build/main data/trillion-processed.csv 1 200 200 1750
srun ./build/main data/trillion-processed.csv 1 200 200 1500
srun ./build/main data/trillion-processed.csv 1 200 200 1250
srun ./build/main data/trillion-processed.csv 1 200 200 1000
srun ./build/main data/trillion-processed.csv 1 200 200 800
srun ./build/main data/trillion-processed.csv 1 200 200 600
srun ./build/main data/trillion-processed.csv 1 200 200 400
srun ./build/main data/trillion-processed.csv 1 200 200 100
