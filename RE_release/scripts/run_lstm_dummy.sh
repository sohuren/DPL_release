#!/bin/bash

LOCAL_HOME=`pwd`/../  # where the theano_src directory reside

cd ${LOCAL_HOME}

DATA_DIR=${LOCAL_HOME}/data # The data directory
PP_DIR=${LOCAL_HOME}/results/Nary_param_and_predictions_olddata_dummy # The directory for the prediction files
OUT_DIR=${LOCAL_HOME}/results/Nary_results_olddata_dummy  # The log output dirctory

mkdir $PP_DIR
mkdir $OUT_DIR


# for the meaning of those parameters, please see the python file
THEANO_FLAGS=mode=FAST_RUN,device=$3,floatX=float32,nvcc.flags=-use_fast_math,exception_verbosity=high python theano_src/lstm_RE.py --setting run_single_corpus --data_dir ${DATA_DIR}/drug_gene_var_dummy/ --emb_dir ${DATA_DIR}/glove/glove.6B.100d.txt --total_fold 7 --dev_fold $1 --test_fold $5 --num_entity 3 --circuit $2 --batch_size 8 --lr 0.02 --lstm_type_dim 2 --content_file sentences_3nd_f.$4 --dependent_file graph_arcs --parameters_file ${PP_DIR}/all_triple_best_params_$2.$4.cv$5.lr0.02.bt8 --prediction_file ${PP_DIR}/all_triple_$2.$4.cv$5.predictions --factor_set $4 > ${OUT_DIR}/all_triple.accuracy.$2.$4.cv$5.lr0.02.bt8.noName
