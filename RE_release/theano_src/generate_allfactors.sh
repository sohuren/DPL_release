#!/bin/bash

# supervision level 3
factor_set=3
# i: cross validation folder i
for i in {0..4}; do
   datadir=../data/drug_gene_var_dummy/${i}
   sbatch -p contrib-cpu generate_factor.sh ${datadir} ${factor_set} 3 # 3 means drug gene var contains 3 mentions
done

# supervision level 2
factor_set=2
# i: cross validation folder i
for i in {0..4}; do
   datadir=../data/drug_gene_var_dummy/${i}
   sbatch -p contrib-cpu generate_factor.sh ${datadir} ${factor_set} 3 # 3 means drug gene var contains 3 mentions 
done

# supervision level 1
factor_set=1
# i: cross validation folder i
for i in {0..4}; do
   datadir=../data/drug_gene_var_dummy/${i}
   sbatch -p contrib-cpu generate_factor.sh ${datadir} ${factor_set} 3 # 3 means drug gene var contains 3 mentions
done

