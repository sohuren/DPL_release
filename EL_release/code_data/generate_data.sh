#!/bin/bash

source ~/.bashrc

data_type=$1
source=pubmed_parsed
length=$2
length_train=$3
input_split=$5

#if [ ! -f data/${source}/${data_type}_${input_split}/${4}/hard/train_${length_train}.pkl ] || [ ! -f data/${source}/${data_type}_${input_split}/${4}/hard/test_${length}.pkl ] || [ ! -f data/${source}/${data_type}_${input_split}/${4}/hard/validation_${length}.pkl ]; then

python generate_dataset.py \
        --input_clinlical_matching data/${source}/pubmed_parsed_splits_dict_${input_split}.pkl \
        --ouput_train data/${source}/${data_type}_${input_split}/${4}/hard/train_${length_train}.pkl \
        --ouput_test data/${source}/${data_type}_${input_split}/${4}/hard/test_${length}.pkl \
        --ouput_val data/${source}/${data_type}_${input_split}/${4}/hard/validation_${length}.pkl \
        --entity_type ${data_type} \
        --min_length ${length} \
        --html_file data/${source}/${data_type}_${input_split}/${4}/hard/sample_${length_train}.html \
	--min_length_train ${length_train} \
        --cv ${4}
#fi
