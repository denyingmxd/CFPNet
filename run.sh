#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
#
python train.py @configs/train_deltar_change_embedding_no_clip_grad_hist_encoder_optimized_10x_cvxt27.txt
python evaluate_all.py @configs/train_deltar_change_embedding_no_clip_grad_hist_encoder_optimized_10x_cvxt27.txt
python evaluate_all.py @configs/train_deltar_change_embedding_no_clip_grad_hist_encoder_optimized_10x_cvxt27.txt --test_dataset nyu


