!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

在现有的模型结果基础上，只需要依次执行：

python essemble.py

python complete_empty.py

就可以得到最终的提交结果。

提交文件位于：

./final_results/result.json

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



以下为代码的执行介绍，模型方案部分请参见评测论文：

# 主要思路

我们利用了评测论文中所提出的模型来完成模型的主要任务。

除评测论文中的模型结构外，我们还利用了以下技术点来提升模型的预测能力：

1. 利用开源工具 LTP 3.4 进行分词、词性、ner和依存句法分析，将这些handcrafted标签embedding化，作为模型预测的特征

2. 利用UER-py发布的RoBERTa模型，作为基础预训练模型。下载网址为:

BERT pretrained on mixed large Chinese corpus (bert-large 24-layers): https://share.weiyun.com/5G90sMJ

3. 为了有效利用全部有标和无标数据，在A榜和B榜的全部数据集上做继续预训练，训练轮数为15000

4. 为了有效利用无标数据（测试集数据），我们在真标数据上训练多个模型后，在测试集上打上伪标签，将伪标签也参与模型训练

5. 训练多个模型，进行模型集成

6. 一些针对数据集的后处理规则，主要包括：限制预测结果的论元长度；按投票数目来去除多余的触发词；去除重复等



# 执行步骤

那么，执行步骤可以描述为：

1. 准备预训练语言模型

BERT pretrained on mixed large Chinese corpus (bert-large 24-layers): https://share.weiyun.com/5G90sMJ

放在 /models 文件夹

2. 准备LTP人工特征的生成

放在 /datasets/hand_feature/ltp/ 文件夹下
已包含处理后的文件

3. 在A榜和B榜的全部数据上，进行RoBERTa的继续预训练，训练轮数为15000

模型放在/models 文件夹下
继续预训练好的模型的下载链接：

4. 用事件检测的结果, 给事件抽取模型打标签

```
cd datasets
python label_cls_results_to_extraction_data.py
cd ..
```

4. B榜数据集进行交叉检验式的划分，按5折交叉检验划分成5份

```
cd datasets
python gen_data_spt_ids.py
python transform_5spt_data_to_train_format.py
cd ..
```

5. 在这5个数据集上进行模型训练，然后将模型在测试集上分别进行结果预测，得到5个模型结果，并集成测试结果

参见 run.bash

6. 利用预测的测试集结果，为测试集打上伪标签

```
cd datasets
combine_fake_data_to_train_truth.py
cd ..
```

7. 对训练集做10折数据划分（包括步骤1）中的5个和新划分的5个），将伪标签的测试集与10个训练集合并，分别训练新的10个模型

参见 run.bash

8. 对新的10个模型和步骤2）中的5个模型结果进行模型集成，共15个模型，投票阈值为11，得到最终的预测结果

```
python essemble.py
python complete_empty.py

```

最终结果位置在：

./final_results/result.json





ps. 由于服务器资源的限制和比赛时间的紧张，我们参与集成的模型会有一些出入，但是主要思路和上述相同。

具体的，参与集成的模型的包括：

1. 步骤2）中的5个模型

/results/model_spt1_result_15000
/results/model_spt2_result_15000
/results/model_spt3_result_15000
/results/model_spt4_result_15000
/results/model_spt5_result_15000

2. 步骤5）的10折数据只跑了模型spt1和spt3~10，共9个模型（漏掉的spt2是因为GPU紧张没有来得及训练），预测结果位于：

/results/model_spt3_result_fakeb_hard
/results/model_spt4_result_fakeb_hard
/results/model_spt5_result_fakeb_hard
/results/model_spt6_result_fakeb_hard
/results/model_spt7_result_fakeb_hard
/results/model_spt8_result_fakeb_hard
/results/model_spt9_result_fakeb_hard
/results/model_spt10_result_fakeb_hard

3. 另外，我们还利用a榜测试集的伪标签的数据，同样的步骤在spt1上训练模型（spt2~10亦可，但太耗时我们只用了spt1），得到了一个预测结果：

/results/model_spt1_result_fakea


