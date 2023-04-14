# 中文个性化语音合成

中文个性化语音合成使用的是目前典型的解码合成器方案来完成语音音色的模拟



# TODO list

-   [x] 初始化项目，统一文件风格
-   [ ] 完善项目工作流，能一键运行
-   [ ] 设计交互界面
-   [ ] 打包成可使用的应用
-   [ ] 绘制项目逻辑图
-   [ ] 写项目测试（选）
-   [ ] 尝试docker部署环境（选）

# 一、数据预处理方法（100data-preprocessing）

## （一）训练数据集的预处理

### 1、中文标点符号处理

对于中文标点符号，只保留'，。？！'四种符号，其余符号按照相应规则转换到这四个符号之一。

### 2、中文到拼音转换

利用字拼音文件和词拼音文件，实现中文到拼音转换，能有效消除多音字的干扰。具体步骤如下：       

1. 对于每个句子中汉字从左到右的顺序，优先从词拼音库中查找是否存在以该汉字开头的词并检查该汉字后面的汉字是否与该词匹配，若满足条件，直接从词库中获取拼音，若不满足条件，从字拼音库中获取该汉字的拼音。     
2. 对于数字(整数、小数)、`ip`地址等，首先根据规则转化成文字，比如整数2345转化为二千三百四十五，再转化为拼音。
3. 由于输入是文字转化而来的拼音序列，所以在合成阶段，允许部分或全部的拼音输入。    

### 3、分词模型

## （二）`TacotronV2`训练数据集的预处理

## （三）`wavernn`、`melgan`训练数据集的预处理

# 二、语音个性化(speaker adaptive)模型训练（`train-pipline`）

[TactronV2](https://github.com/Rayhane-mamah/Tacotron-2)支持finetune，固定decoder层前的参数(embedding层、CHBG、encoder层等)，用新数据集(数据量很少)训练从checkpoint中恢复的模型，达到`speaker adpative`的目的。

在数据预处理的基础上调用TacotronV2网络迁移指定模拟说话人的音色，生成指定说话人的TacotronV2微调模型

# 三、模型使用（推理）（inference-pipeline）

1、使用指定说话人的微调模型生成文本的中间表示——Mel频谱图；

2、处理Mel频谱图，尝试了两种处理方案，对比之后发现`melgan`的处理效果较好，语音自然、噪点较少。

## `melgan`使用说明

```markdown
usage: melgan_inference.py [-h] [-i FOLDER] [-o SAVE_PATH] [-m LOAD_PATH]
                           [--mode MODE] [--n_samples N_SAMPLES]

optional arguments:
  -h, --help            show this help message and exit
  -i FOLDER, --folder FOLDER
                        输入音频文件的目录路径
  -o SAVE_PATH, --save_path SAVE_PATH
                        输出生成语音的目录路径
  -m LOAD_PATH, --load_path LOAD_PATH
                        模型路径
  --mode MODE           模型模式
  --n_samples N_SAMPLES
                        需要实验多少个音频
```



# 资源链接与参考

## 语音合成参考方案

[TacotronV2 + WaveRNN语音合成](https://github.com/lturing/tacotronv2_wavernn_chinese)

## 注意力机制改良

[location-relative attention mechanisms for robust long-form speech synthesis](https://arxiv.org/pdf/1910.10288)

> 由于[TacotornV2]()中采用的注意力机制是[Location sensitive attention](https://github.com/lturing/tacotronv2_wavernn_chinese/blob/master/tacotron/models/location_sensitive_attention.py)，对长句子的建模能力不太好，尝试了以下注意力机制：    

* [Guassian mixture attention](https://github.com/lturing/tacotronv2_wavernn_chinese/blob/master/tacotron/models/gmm_attention.py)
* [Discretized Graves attention](https://github.com/lturing/tacotronv2_wavernn_chinese/blob/master/tacotron/models/graves_attention.py)
* [Forward attention](https://github.com/lturing/tacotronv2_wavernn_chinese/blob/master/tacotron/models/forward_attention.py)

> 由于语音合成中的音素(拼音)到声学参数(Mel频谱)是从左到右的单调递增的对应关系，特别地，在合成阶段，对[forward attention](https://github.com/lturing/tacotronv2_wavernn_chinese/blob/master/tacotron/models/forward_attention.py#L171)中的alignments的计算过程的特殊处理，能进一步提高模型对长句子的语音合成效果，以及控制语速。
