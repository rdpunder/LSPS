#!/bin/bash
#Set job requirements
#SBATCH -N 2 --tasks-per-node 128 -t 23:00:00 -p rome
#SBATCH --mail-type=FAIL,END
#SBATCH --ear=off

# Load modules
source ~/JHS_installations/venvs/LSPS/bin/activate
module load 2023
module load Lumerical/2023-R2.3-OpenMPI-4.1.5

#Run program
mpirun -n 256 python 01PowerMain_NS5_L_C20_Calc.py "TMPDIR"/input_dir "$TMPDIR"/output_dir