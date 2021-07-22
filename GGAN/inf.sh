#!/usr/bin/env bash

cd /home/hlcv_team019/Text2Image/GGAN/

/home/hlcv_team019/miniconda3/envs/myenv1/bin/python inference.py --dataset birds --num_workers 10 --gen_dir="models/GGAN_BIRDS_EMB2/GGAN_BIRDS_EMB2_gen_600.pth" --experiment_name "GGAN_BIRDS_EMB2_gen_600"
