#!/bin/bash

data_type=gene # gene mention, also can choose disease, drug etc
source=pubmed_parsed # pubmed dataset
length=5  # length heuristic for validation
length_train=3 # length heuristic for train


# j: cross-validation, if no cross validation, just set j=0

# all hard em
for i in {0..0}; do
mkdir data/${source}/${data_type}_${i}/
done

for i in {0..0}; do
for j in {0..0}; do
mkdir data/${source}/${data_type}_${i}/${j}/
done
done

for i in {0..0}; do
for j in {0..0}; do
mkdir data/${source}/${data_type}_${i}/${j}/hard
sbatch -p contrib-cpu -C highmem generate_data.sh ${data_type} ${length} ${length_train} ${j} ${i} 
done         
done

# all soft em with full feature set
for i in {0..0}; do
for j in {0..0}; do
for k in {1..3}; do # k=1,2,3 represent different supervision level
mkdir data/${source}/${data_type}_${i}/${j}/soft_featureset_${k}
sbatch -p contrib-cpu -C highmem generate_data_soft.sh ${data_type} ${length} ${length_train} ${j} ${k} ${i}
done
done
done 
