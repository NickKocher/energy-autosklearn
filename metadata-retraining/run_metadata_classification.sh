#!/bin/bash

#SBATCH --job-name=asklearn-meta
#SBATCH -a 0-0%256
#SBATCH --time=48:00:00
#SBATCH --partition=KathleenE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive

#SBATCH --mem=15G
#SBATCH --output=asklearn-meta_%A-%a.out
cd /home/kocher/energy-autosklearn/metadata-retraining
# Run the metadata job
source /home/kocher/energy-autosklearn/setup-env.sh
for ((i=1; i<2; i++)) ; do
line=$((${SLURM_ARRAY_TASK_ID}*2+i));
export BENCHMARKDIR=/home/kocher/energy-autosklearn/metadata-retraining/run_2;
export ITERATIONS=10;
mkdir -p $BENCHMARKDIR;
command=$(sed -n "${line}p" /home/kocher/energy-autosklearn/metadata-retraining/metadata_commands.txt);
srun $command;
done