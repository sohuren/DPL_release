#!/bin/sh

# comments are the same for distant supervision setting

data_type=gene
source=pubmed_parsed
length=5
length_train=3

source ~/.bashrc

mkdir data/${source}/${data_type}_result/0

for j in {0..0}; do
        
dataroot=data/${source}/${data_type}_${j}/0/soft_featureset_${1}
resultroot=data/${source}/${data_type}_result/0/
python baseline_rnn.py \
        --dataroot ${dataroot} \
        --train_data train_${length_train}.pkl \
        --val_data validation_anno_full.pkl \
        --test_data test_${length}.pkl \
        --visulization_html ${resultroot}/prediction_soft_featureset_${1}_${length_train}_learngraph.html \
        --confidence_html ${resultroot}/confidence_soft_featureset_${1}_${length_train}_learngraph.html \
        --prediction_file ${resultroot}/prediction_${length_train}_${1} \
        --save_path ${resultroot}/model_soft_featureset_${1}_${length_train}_learngraph.pkl  \
        --stage "M" \
        --epochs 1 \
        --multiple_M 5 \
        --learn_graph > ${resultroot}/soft_featureset_${1}_${length_train}_learngraph_M_full.log
done
