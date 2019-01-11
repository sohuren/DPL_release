#!/bin/sh

source ~/.bashrc

python ./theano_src/message_passing.py --data_path $1 --out_path $2
