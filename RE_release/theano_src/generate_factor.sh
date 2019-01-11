#!/bin/bash

source ~/.bashrc

# --num_entities: number of entities. For drug gene var, this is always 3
# if existed the file, then skip 
#if [ ! -f $1/factorgraph_${2}.pkl ]; then
python generate_factorgraph.py --raw_file $1/sentences_raw  --graph_file $1/graph_arcs --result_file $1/factorgraph_${2}.pkl --factor_set $2 --num_entities $3 --input_probability_file $1/sentences_2nd --input_gene_prob_file $1/sentences_gene_prediction --output_probability_file $1/sentences_3nd_f.$2
#fi
