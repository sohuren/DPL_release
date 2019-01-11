#!/bin/bash

source ~/.bashrc

# run the distant supervision baseline, with the length + Distant Supervision as the supervision

bash run_experiment_DS.sh # table.5, DS setting

# run the DPL with different level of supervision

bash run_experiment_DPL.sh 1 # (+ data programming on mention level), table.5, DS + DP setting
bash run_experiment_DPL.sh 3 # (+ joint inference), table.5, DS + DP + JI setting
