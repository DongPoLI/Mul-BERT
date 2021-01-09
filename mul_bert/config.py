# -*- coding: utf-8 -*-
"""
@Time   : 2021/1/8 下午8:08
@Author : Li Shenzhen
@File   : config.py
@Software:PyCharm
"""


class Config:
    # pretrained_weights = "bert-base-uncased"
    pretrained_weights = "bert-large-uncased"

    # 模型路径
    # checkpoint_dir = "mul_bert_model/mul_bert_base_8943_checkpoint.pth.tar"
    # checkpoint_dir = "mul_bert_model/mul_bert_large_9005_checkpoint.pth.tar"
    checkpoint_dir = "mul_bert_model/mul_bert_large_9028_checkpoint.pth.tar"


config = Config()