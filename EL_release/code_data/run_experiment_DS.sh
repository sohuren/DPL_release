#!/bin/bash

data_type=gene #gene
source=pubmed_parsed # data split
length=5 # length for validation
length_train=3 # length for train set

# should be very fast since the file is small
source ~/.bashrc

# make dir for saving the result&log
mkdir data/${source}/${data_type}_result/0


for j in {0..0}; do
# j represent the data split 
dataroot=data/${source}/${data_type}_${j}/0/hard
resultroot=data/${source}/${data_type}_result/0/ #3
python baseline_rnn.py \
        --dataroot ${dataroot} \
        --train_data train_${length_train}.pkl \
        --val_data validation_anno_full.pkl \
        --test_data test_${length}.pkl \
        --visulization_html ${resultroot}/prediction_hard_${length_train}.html \
        --confidence_html ${resultroot}/confidence_hard_${length_train}.html \
        --save_path ${resultroot}/model_hard_${length_train}.pkl  \
        --prediction_file ${resultroot}/prediction_${length_train}_hard \
        --hard_em True \
        --epochs 5 > ${resultroot}/hard_${length_train}_full.log
done
