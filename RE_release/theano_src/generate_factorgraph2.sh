#!/bin/bash

source ~/.bashrc
python ./theano_src/generate_factorgraph_chunks.py --input_data $1 --ouput_data $2 --input_entitytype_freq $3 --input_entity_ctx_freq $4 --factor_set $5
 