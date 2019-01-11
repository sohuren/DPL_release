#!/bin/sh

# submission script for slurm cluster
sbatch -p contrib-cpu ./theano_src/submit.sh $1 $2
