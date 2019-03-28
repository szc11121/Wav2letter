# Wav2Letter with Tensorflow

Wav2Letter是由Facebooks AI Research (FAIR)发表的一个语音识别模型，具体的论文可以在这里看到 [paper](https://arxiv.org/pdf/1609.03193.pdf)。

第二代Wav2Letter的论文 [paper](https://arxiv.org/abs/1712.09444)。 使用了Gated Convnets代替了普通的Convnets。



## 与论文的区别

* 使用了 CTC Loss
* 使用了 Greedy Decoder 

## Getting Started

## Requirements

```bash
pip install -r requirements.txt
```


## Data

使用 [Google Speech Command Dataset](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data). 这是一个包含许多1秒钟音频文件的轻量级数据集。

### download data

将文件下载并解压到 `./speech_data/speech_commands_v0.01/` 文件夹下。

### Prepare data


```bash
python Wav2Letter/data.py
```
`data.py` 文件会对解压好的数据进行预处理，生成模型训练所需要数据和标签，保存成 `./speech_data` 里的`x.npy` ， `y.npy` ，`x_length.npy`

## Train

```bash
python train.py --batch_size=256 --epochs=1000
```
训练过程中会显示训练集的LOSS，每个epoch会显示测试集的loss并且使用测试集的第一个样本进行输出测试。