# 主要思路

将原始文本通过RoBERTa，对文本进行事件多分类。利用多模型集成的方式，提升模型的最终预测结果



# 路径配置

比赛数据放置在datasets/data中

将预训练模型放置在pretrained_model/chinese_roberta_wwm_large_ext_pytorch，需包含三个文件：
1. BERT词典：pretrained_model/chinese_roberta_wwm_large_ext_pytorch/vocab.txt
2. BERT结构：pretrained_model/chinese_roberta_wwm_large_ext_pytorch/bert_config.json
3. BERT模型：pretrained_model/chinese_roberta_wwm_large_ext_pytorch/pytorch_model.bin

预训练语言模型下载地址：RoBERTa-wwm-ext-large, Chinese
http://pan.iflytek.com/#/link/9B46A0ABA70C568AAAFCD004B9A2C773
密码43eH



# 执行步骤

1. 执行数据预处理

python data_preprocess.py

注：会生成csv文件放在datasets/pred_trans中

2. 生成用来训练的10个数据集划分id

```
cd ./datasets
python gen_data_spt_ids.py
cd ..
```

3. 根据数据集划分id，划分本地训练集/验证集

python split_data_by_ids.py

注：所用的预先的设置，包含5折交叉检验式划分，分别位于datasets/spt1/data_ids.json中；生成的pkl文件会在datasets/spt1中

3. 执行训练

```
CUDA_VISIBLE_DEVICES=1 \
nohup \
python -u run_bert_sigmoid.py --do_train --save_best \
    --train_data_path datasets/spt1/trans_train.train.pkl \
    --valid_data_path datasets/spt1/trans_train.valid.pkl \
    --arch spt1_trans_sig --type trans \
    > logs/spt1.trans_sig &
```

4. 执行测试

```
CUDA_VISIBLE_DEVICES=1 \
nohup \
python -u run_bert_sigmoid.py --do_test --save_best \
    --train_data_path datasets/spt1/trans_train.train.pkl \
    --valid_data_path datasets/spt1/trans_train.valid.pkl \
    --test_raw_data_path datasets/pred_trans/online_sample.csv \
    --arch spt1_trans_sig --type trans \
    --results_output results/sig/spt1.trans.cls \
    > spt1.trans_test_sig &
```

输出文件位置为： /results/sig/spt1.trans.cls

5. 集成结果

python essemble.py

对训练的10个模型，合并预测结果

输出文件位置为： /results/essemble/trans_5.cls

并作为EventExtraction任务的输入
