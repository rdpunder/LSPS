#!/bin/bash
#Set job requirements
#SBATCH -N 1 --tasks-per-node 128 -t 1:00:00 -p rome
#SBATCH --ear=off

# Load modules
source ~/JHS_installations/venvs/LSPS/bin/activate
module load 2023
module load Lumerical/2023-R2.3-OpenMPI-4.1.5

#Run program
mpirun -n 128 python InflationScoreCalcMain_C_h24.py "TMPDIR"/input_dir "$TMPDIR"/output_dir