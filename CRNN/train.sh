#!/bin/bash
set -x
CUDA_VISIBLE_DEVICES=2 ./train.py --adadelta --trainRoot dataset_digit_20/trainset.lmdb --valRoot dataset_digit_20/valset.lmdb --cuda --saveInterval 10000 --valInterval 10000 --expr_dir expr_digit_20 | tee run.log
