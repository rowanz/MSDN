#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 python train_hdn.py \
    --resume_training --resume_model ./output/trained_model/HDN_1_iters_alt_normal_I_LSTM_with_bias_with_dropout_0_5_nembed_256_nhidden_512_with_region_regression_resume_SGD_best.h5 \
    --dataset_option=normal  --MPS_iter=1 \
    --caption_use_bias --caption_use_dropout \
    --rnn_type LSTM_normal --evaluate --mode sgcls