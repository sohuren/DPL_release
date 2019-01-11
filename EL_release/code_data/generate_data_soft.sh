#!/bin/bash

source ~/.bashrc

data_type=$1 # gene
source=pubmed_parsed # data source
length=$2 # length
length_train=$3 # length for train set
input_split=$6 # cross validation

#if [ ! -f data/${source}/${data_type}_${input_split}/${4}/soft_featureset_${5}/train_${length_train}.pkl ] || [ ! -f data/${source}/${data_type}_${input_split}/${4}/soft_featureset_${5}/test_${length}.pkl ] || [ ! -f data/${source}/${data_type}_${input_split}/${4}/soft_featureset_${5}/validation_${length}.pkl ]; then 
python generate_dataset.py \
        --input_clinlical_matching data/${source}/pubmed_parsed_splits_dict_${input_split}.pkl \
        --ouput_train data/${source}/${data_type}_${input_split}/${4}/soft_featureset_${5}/train_${length_train}.pkl \
        --ouput_test data/${source}/${data_type}_${input_split}/${4}/soft_featureset_${5}/test_${length}.pkl \
        --ouput_val data/${source}/${data_type}_${input_split}/${4}/soft_featureset_${5}/validation_${length}.pkl \
        --entity_type ${data_type} \
        --min_length ${length} \
        --html_file data/${source}/${data_type}_${input_split}/${4}/soft_featureset_${5}/sample_${length_train}.html \
	--min_length_train ${length_train} \
        --cv ${4} \
        --hard_em 0 \
        --factor_set ${5}
#fi
