# 个性化语音合成

*   又叫做语音克隆
*   特别致谢：这个项目是在[`lturing`的开源项目](https://github.com/lturing/tacotronv2_wavernn_chinese)项目的基础上进行的，在数据工程、模型结构、训练、部署等方面做了不同程度的改进。感谢作者、感谢互联网分享。
*   原项目解读：
    *   master分支：采用开源语音数据集标贝(女声)，得到预训练TacotronV2模型
    *   adaptive分支：可配置超参数，从而利用指定说话人的少量数据微调预训练的Tacotron模型，从开源语音数据集thchs30中选择了D8(男声)，一共250句

## 项目思路

*   效率工具
*   组合模型
    *   分词模型
    *   编解码器

## 典型处理流程

*   数据采集与预处理
*   模型微调
*   模型部署使用

## 参考资料

*   #TODO





