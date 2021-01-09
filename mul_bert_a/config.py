# -*- coding: utf-8 -*-
"""
@Time   : 2021/1/8 下午8:08
@Author : Li Shenzhen
@File   : config.py
@Software:PyCharm
"""


# 修改此配置文件；
class Config:
    # pretrained_weights = "bert-base-uncased"
    pretrained_weights = "bert-large-uncased"

    # 模型路径
    # checkpoint_dir = "mul_bert_a_model/mul_bert_a_base_8884_checkpoint.pth.tar"
    checkpoint_dir = "mul_bert_a_model/mul_bert_a_large_8955_checkpoint.pth.tar"


config = Config()