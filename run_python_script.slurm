#!/bin/bash
#SBATCH --job-name=DarkSage_analysis

#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --nodes=1

module load scipy/1.0.0-python-3.6.4
module load gsl/2.4
module load astropy/2.0.3-python-3.6.4
module load numpy/1.14.1-python-3.6.4
module load openmpi/3.0.0
module load gcc/6.4.0
module load matplotlib/2.2.2-python-3.6.4
module load mpi4py/3.0.0-python-3.6.4

PROGNAME="/fred/oz042/rdzudzar/python/explore_simulations/FunctionsDarkSage.py"

time mpirun -np 1 python3 $PROGNAME
