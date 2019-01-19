# Text Classification (Sentence level)

Experiments on sentence classification. Task-Independent Model Wrapper. 


## Task-independent Model Wrapper

A supervised learning project commonly has the following modules:

* Dataset module (including data-preprocessing, data-batching)
* Model defining, or graph defining module
* Training, validation, prediction procedures

The train/valid/pred procedures for supervised learnings are much the same for different tasks, so I tried to implement the model wrapper to be task-independent. Then one can focus on data preprocessing and model construction jobs.

## Models

* CNN
* CNN-RNF
* RNN & Attention-Pooling
* Multi-head Self-Attention

## Description

To run this repo:

1, python data_set.py

2, python script_train_and_eval.py --model=cnn

3, python script_predict.py --model=cnn

</br>

Requiring package jieba for token-segmentation:

pip install jieba


## References

1, Convolutional Neural Networks for Sentence Classification ( https://arxiv.org/abs/1408.5882 )

2, Convolutional Neural Networks with Recurrent Neural Filters ( https://arxiv.org/abs/1808.09315 )

3, Neural Machine Translation by Jointly Learning to Align and Translate ( https://arxiv.org/abs/1409.0473 )

4, Attention Is All You Need ( https://arxiv.org/abs/1706.03762 )

5, https://github.com/gaussic/text-classification-cnn-rnn

6, https://github.com/bloomberg/cnn-rnf

...


