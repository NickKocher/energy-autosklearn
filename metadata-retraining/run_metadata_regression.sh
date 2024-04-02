#!/bin/bash

#SBATCH --job-name=asklearn-meta2
#SBATCH -a 0-110
#SBATCH --time=120:00:00
#SBATCH --partition=Kathleen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --mem=15G
#SBATCH --output=asklearn-meta_%A-%a.out

# Run the metadata job
source /home/kocher/energy-autosklearn/setup-env.sh

for ((i=1; i<4; i++)) ; do
line=$((${SLURM_ARRAY_TASK_ID}*3+i));
command=$(sed -n "${line}p" /home/kocher/energy-autosklearn/metadata-no-energy/metadata_commands.txt);
srun $command &;
done
