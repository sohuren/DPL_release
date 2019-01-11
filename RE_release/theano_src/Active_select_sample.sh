#!/bin/bash

source ~/.bashrc

# for parameters on those files, see the python file
python theano_src/active_select_sample.py \
    --raw_file $1 --anno_file $2 --input_probability_file $3 --input_previous_sample_file $4 --input_previous_graph_file $5 \
    --input_gene_prob_file $6 --graph_file $7 --factor_set $8 --num_entities $9 \
    --output_samples_file ${10} --output_graph_file ${11}
