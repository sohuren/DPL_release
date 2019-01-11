#!/bin/bash

# submission script for slurm cluster 
sbatch -p contrib-cpu generate_factorgraph.sh $1 $2 $3 $4 $5 
