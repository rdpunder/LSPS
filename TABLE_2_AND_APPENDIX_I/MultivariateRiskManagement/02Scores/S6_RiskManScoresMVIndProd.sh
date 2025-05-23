#!/bin/bash
#Set job requirements
#SBATCH -N 6 --tasks-per-node 192 -t 23:59:00 -p genoa
#SBATCH --ear=off

# Load modules
source ~/JHS_installations/venvs/LSPS/bin/activate
module load 2023
module load Lumerical/2023-R2.3-OpenMPI-4.1.5

#Run program
mpirun -n 1152 python RiskManMainMVIndProd.py "TMPDIR"/input_dir "$TMPDIR"/output_dir