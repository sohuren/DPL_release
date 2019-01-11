#!/bin/bash

source ~/.bashrc

# use GraphLSTM as the DL module
bash run_lstm_dummy.sh 5 GraphLSTMRelation gpu 3 6 # 3 is the indicator of supervision level, contains DS+DP+JI (row 3 in table 2) 
# use biLSTM as the DL module
#bash run_lstm_dummy.sh 5 LSTMRelation gpu 3 6

bash run_lstm_dummy.sh 5 GraphLSTMRelation gpu 1 6 # 1 is the indicator of supervision level, contains DS + DP on entity level
#bash run_lstm_dummy.sh 5 LSTMRelation gpu 1 6
bash run_lstm_dummy.sh 5 GraphLSTMRelation gpu 2 6 # 2 is the indicator of supervision level, contains DS + DP on entity level + mention pair level
#bash run_lstm_dummy.sh 5 LSTMRelation gpu 2 6
