## 2020  CCF-BDCI 大数据与计算智能大赛：大数据时代的Serverless工作负载预测

---
### 作者与代码简介
鱼丸粗面(zhuoyin94@163.com)，20201021开赛。源码主要包括数据预处理，特征工程与模型训练部分。特征工程依据模型不同进行了两部分工作：第一种是针对多输出的神经网络的特征；第二种是针对lgb的特征。其中组织Target的时候针对Target的越界的问题进行了阈值的滤除。

---
### 系统环境与主要依赖packages
- 系统环境: Ubuntu 18.04 LTS
- GPU: GeForce GTX 1650
- CPU: Intel® Core™ i7-9750H CPU @ 2.60GHz × 12
- RAM: 64GB

- python: 3.6.9
- sklearn: 0.23.1
- pandas: 1.0.4
- tensorflow-gpu: 2.1.0

---
### 文件结构
主要文件目录结构如下：
- **.//submissions//**: 用于存储模型预备提交的结果。
- **.//submission_oof//**: 用于存储模型训练过程中每一fold的out of fold预测结果，方便后续进行stacking工作。
- **.//data//**: 用于存储原始官方数据。
- **.//cached_data//**: 用于存储预处理后的数据与一些临时数据。

---
### 代码介绍
主要工作文件介绍如下：
- **preprocessing.py**: 用于对原始数据进行预处理。
- **run_feat_engineering_nn.py**: 针对多任务神经网络的特征工程。
- **run_feat_engineering_lgb.py**: 针对lightgbm的特征工程。
- **model_boosting_lgb.py**: 读取lgb特征工程结果，并训练lgb模型。
- **model_multitask_nn.py**: 读取multitask_nn特征工程结果，并训练NN模型。
- **utils.py**: 辅助函数文件。





