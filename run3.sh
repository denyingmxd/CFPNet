#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3


python evaluate_all.py @configs/train_deltar_change_embedding_no_clip_grad_hist_encoder_optimized_10x_combine16.txt
