#!/bin/bash
#Set job requirements
#SBATCH -N 1 --tasks-per-node 192 -t 3:00:00 -p genoa
#SBATCH --mail-type=FAIL,END
#SBATCH --ear=off

# Load modules
source ~/JHS_installations/venvs/LSPS/bin/activate
module load 2023
module load Lumerical/2023-R2.3-OpenMPI-4.1.5

#Run program
mpirun -n 192 python 01SizeMain_Calc.py "TMPDIR"/input_dir "$TMPDIR"/output_dir