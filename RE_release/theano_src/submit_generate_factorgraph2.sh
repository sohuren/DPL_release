#!/bin/bash

# submission script for ttic slurm cluster[slurm.ttic.edu]
sbatch -p contrib-cpu ./theano_src/generate_factorgraph2.sh $1 $2 $3 $4 $5 
