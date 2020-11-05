### 2020  CCF-BDCI 大数据与计算智能大赛：大数据时代的Serverless工作负载预测

---
### 队伍简介
队名：我们的队伍向太阳，队长：鱼丸粗面(zhuoyin94@163.com)。20201021开赛。

---
### 系统环境与主要依赖packages
- 系统环境: Ubuntu 20.04 LTS
- GPU: Quadro GV100(32G) × 2
- CPU: Intel® Xeon(R) Gold 6138 CPU @ 2.00GHz × 80
- RAM: 256G
- python: 3.6.9
- tensorflow-gpu: 2.1.0
- gensim: 3.8.1
- sklearn: 0.23.1
- pandas: 1.0.4

---
### 文件结构
文件目录如下：
- **.//plots//**: 用于存储神经网络训练过程中每一fold的loss curve与accuracy curve，并采用S-G滤波进行平滑。
- **.//submissions//**: 用于存储当前神经网络的运行结果。
- **.//submission_oof//**: 用于存储神经网络训练过程中每一fold的out of fold预测结果。
- **.//data//**: 用于存储原始官方数据。
- **.//cached_data//**: 用于存储预处理后的数据与一些临时数据。

