# Text Classification (Plain Sequence)

Experiments on sentence classification. Task-Independent Model Baseboard/Wrapper. 

Single GPU or Multi GPU.


## Models

* CNN
* CNN-RNF
* RNN & Attention-Pooling
* Multi-head Self-Attention
* Capsule Network

## Description

To run this repo:

1, python script_data.py

2, python script_runner.py --model_tag=cnn --mode=train

3, python script_runner.py --model_tag=cnn --mode=predict


</br>


To specifiy gpu to be used:

2', python script_runner.py --model_tag=cnn --mode=train --gpu=0

2'', python script_runner.py --model_tag=cnn --mode=train --gpu=0,1


</br>

Using package jieba for token-segmentation:

pip install jieba

Using package Zeras for model baseboard:

pip install Zeras==0.4.1



## Task-independent Model Baseboard/Wrapper

A supervised learning project commonly has the following modules:

* Dataset module (including data-preprocessing, data-batching)
* Model defining, or graph defining module
* Training, validation, prediction procedures

The train/valid/pred procedures for supervised learnings are much the same for different tasks, so I tried to implement the model wrapper to be task-independent. Then one can focus on data preprocessing and model construction jobs.



## References

1, Convolutional Neural Networks for Sentence Classification ( https://arxiv.org/abs/1408.5882 )

2, Convolutional Neural Networks with Recurrent Neural Filters ( https://arxiv.org/abs/1808.09315 )

3, Neural Machine Translation by Jointly Learning to Align and Translate ( https://arxiv.org/abs/1409.0473 )

4, Attention Is All You Need ( https://arxiv.org/abs/1706.03762 )

5, Dynamic Routing Between Capsules ( https://arxiv.org/abs/1710.09829 )

6, Information Aggregation via Dynamic Routing for Sequence Encoding ( https://arxiv.org/abs/1806.01501 )

7, https://github.com/gaussic/text-classification-cnn-rnn

8, https://github.com/brightmart/text_classification

9, https://github.com/bloomberg/cnn-rnf

...


