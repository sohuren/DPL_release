#!/bin/sh
source ~/.bashrc
if [ "$7" == "False" ]
then
   python generate_traindata.py --input_clinlical_matching $1 --ouput_train $2 --entity_type $3 --min_length_train $4 --input_entitytype_freq $5 --input_entity_ctx_freq $6 --factor_set $8
else
   python generate_traindata.py --input_clinlical_matching $1 --ouput_train $2 --entity_type $3 --min_length_train $4 --input_entitytype_freq $5 --input_entity_ctx_freq $6 --hard_em --factor_set $8
fi
