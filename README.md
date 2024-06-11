
# CCKS-2020 Financial Event Extraction

Source code for CCKS 2020 competition rank-1 paper: [A Joint Learning Framework for the CCKS-2020 Financial Event Extraction Task](https://direct.mit.edu/dint/article/3/3/444/100995/A-Joint-Learning-Framework-for-the-CCKS-2020). 

This paper presents a winning solution for the CCKS-2020 financial event extraction task, where the goal is to identify event types, triggers and arguments in sentences across multiple event types. In this task, we focus on resolving two challenging problems (i.e., low resources and element overlapping) by proposing a joint learning framework, named SaltyFishes. We first formulate the event extraction task as a joint probability model. By sharing parameters in the model across different types, we can learn to adapt to low-resource events based on high-resource events. We further address the element overlapping problems by a mechanism of Conditional Layer Normalization, achieving even better extraction accuracy. The overall approach achieves an F1-score of 87.8% which ranks the first place in the competition.

The original link to the competition is [here](https://www.biendata.xyz/competition/ccks_2020_3/).


# How to run

Please refer to ``EventExtraction\README.md`` and ``EventDetection\README.md``.

The data can be obtained from [here](https://pan.baidu.com/s/1moPhPqLrTIOKGF0-xpj77Q?pwd=7bdj).
Please unzip and place ``ED-datasets.zip`` to ``EventDetection\datasets``, and ``EE-datasets.zip`` to ``EventExtraction\datasets``.

# Award

The Top 1 Winner of CCKS 2020 Competition: Few-shot Cross-domain Event Extraction Competition, Chinese Information Processing Society of China. 2020.

The Technological Innovation Award of CCKS 2020 Competition: Few-shot Cross-domain Event Extraction Competition, Chinese Information Processing Society of China. 2020.

# Citation

If you find this code useful, please cite our work:
```
@article{DBLP:journals/dint/ShengLHGYWHLX21,
  author       = {Jiawei Sheng and
                  Qian Li and
                  Yiming Hei and
                  Shu Guo and
                  Bowen Yu and
                  Lihong Wang and
                  Min He and
                  Tingwen Liu and
                  Hongbo Xu},
  title        = {A Joint Learning Framework for the {CCKS-2020} Financial Event Extraction
                  Task},
  journal      = {Data Intell.},
  volume       = {3},
  number       = {3},
  pages        = {444--459},
  year         = {2021},
  url          = {https://doi.org/10.1162/dint\_a\_00098},
  doi          = {10.1162/DINT\_A\_00098}
}
```
