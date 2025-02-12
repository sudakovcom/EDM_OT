#!/bin/bash
#SBATCH --job-name=fid_male2femaleEDM_4gpu_192nc_discriminator
#SBATCH --output=outputs/out.log
#SBATCH --error=outputs/out.err
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --constraint="type_a|type_b|type_c"

torchrun --standalone --nproc_per_node=4 train.py --outdir=outdir --data1train=datasets/male64train.zip --data2train=datasets/female64train.zip --data1test=datasets/male64test.zip --data2test=datasets/female64test.zip --data1stats=datasets/male64train.npz --data2stats=datasets/female64train.npz --batch=256 --precond=no --batch-gpu=32 --t_iters=1 --f_iters=1