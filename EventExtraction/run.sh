#!/usr/bin/env bash
0. 用事件检测的结果, 给事件抽取模型打标签

python label_cls_results_to_extraction_data.py

1. 训练5个金标模型

CUDA_VISIBLE_DEVICES=0,1 \
nohup \
python3 -u train_v9.py --train_path datasets/spt1/train_format.json \
                   --dev_path datasets/spt1/dev_format.json \
                   --test_path datasets/spt1/dev_format.json \
                   --report_steps 20 \
                   --epochs_num 20 --batch_size 8 --seq_length 400 --encoder bert \
                   --output_model_path ./models_save/v9.2.4_spt1_15000.bin \
                   --pretrained_model_path models/new_model_15000.bin \
                   > logs/spt1_15000.log &

2. 测试5个金标模型

CUDA_VISIBLE_DEVICES=15 \
python3 -u test_v9.py --train_path datasets/spt1/train_format.json \
                   --dev_path datasets/spt1/dev_format.json \
                   --test_path datasets/testing_data/test_format_new.json \
                   --batch_size 1 --seq_length 400 --encoder bert \
                   --output_model_path ./models_save/v9.2.4_spt1_15000.bin \
                   --results_path results/model_spt1_result_15000

3. 训练10个带有B榜伪标签数据的模型

CUDA_VISIBLE_DEVICES=0,1 \
nohup \
python3 -u train_v9.py --train_path datasets/spt1/train_format_b.json \
                   --dev_path datasets/spt1/dev_format.json \
                   --test_path datasets/spt1/dev_format.json \
                   --pretrained_model_path models/new_model_15000.bin \
                   --report_steps 20 \
                   --epochs_num 20 --batch_size 8 --seq_length 400 --encoder bert \
                   --output_model_path ./models_save/v9.2.4_spt1_15000_fakeb_hard.bin \
                   > logs/spt1_15000_fakeb_hard.log &

4. 测试10个带有B榜伪标签数据的模型

CUDA_VISIBLE_DEVICES=1 \
nohup \
python3 -u test_v9.py --train_path datasets/spt1/train_format_b.json \
                   --dev_path datasets/spt1/dev_format.json \
                   --test_path datasets/testing_data/test_format_new.json \
                   --batch_size 1 --seq_length 400 --encoder bert \
                   --output_model_path ./models_save/v9.2.4_spt1_15000_fakeb_hard.bin \
                   --results_path results/model_spt1_result_fakeb_hard \
                   > logs/spt1_test_fakeb_hard.log &

5. 模型集成

python essemble.py

6. 补充没有预测结果的id, 形成提交格式的result.txt

python complete_empty.py
