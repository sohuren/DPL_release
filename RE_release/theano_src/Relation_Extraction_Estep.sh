#!/bin/bash

source ~/.bashrc
python theano_src/Relation_Extraction_Estep.py --data_dir $1 --cv $2 $3 --epoch $4 --factor_set $5
