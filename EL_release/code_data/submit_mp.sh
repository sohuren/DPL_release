#!/bin/sh

# auto submit code for slurm.ttic.edu
sbatch -p contrib-cpu submit.sh $1 $2
