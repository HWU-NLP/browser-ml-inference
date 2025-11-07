#!/bin/bash -l

################# Slurm directives ####################
## Working dir
#SBATCH -D /users/aj2066/sharedscratch/browser-ml-inference
## Environment variables
#SBATCH --export=ALL
## Output and Error Files
#SBATCH -o logs/job-onnx-gbv-%j.output
#SBATCH -e logs/job-onnx-gbv-%j.error
## Job name
#SBATCH -J onnx-gbv
## Run time: "hours:minutes:seconds", "days-hours"
#SBATCH --time=150:00:00
## Memory limit (in megabytes). Total --mem or amount per cpu --mem-per-cpu
#SBATCH --mem=32G
## GPU requirements
#SBATCH --gres=gpu:1
## Specify partition
#SBATCH -p gpu


conda activate onnx
python train_gbv_model.py
