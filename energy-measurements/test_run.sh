#!/usr/bin/bash
# Name the job
#SBATCH --job-name=Kathleen.%j

### Start of Slurm SBATCH definitions
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --partition=KathleenE
# Ask for the maximum memory per CPU
#SBATCH --mem=18000M

# Ask for up to 10 Minutes of runtime
#SBATCH --time=00:20:00



# Declare a file where the STDOUT/STDERR outputs will be written
#SBATCH --output=testslurm.%J


### end of Slurm SBATCH definitions
export BENCHMARKDIR=/home/kocher/energy-autosklearn/energy-measurements/test
export ITERATIONS=5
mkdir -p $BENCHMARKDIR;
### your program goes here (hostname is an example, can be any program)
# `srun` runs `ntasks` instances of your programm `hostname`
source /home/kocher/energy-autosklearn/setup-env.sh
srun --mpi=pmix -n 1 python3 basic_dataset_test.py


