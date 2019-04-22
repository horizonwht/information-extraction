#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time   : 19-4-18 下午5:08
# @Author : wanghuiting
# @File   : test.py
from lib.get_vocab import get_vocab
from lib.get_spo_train import get_p
from bin.p_classification.p_train import main
from bin.so_labeling.spo_train import spo_main
from config.ConfigParameter import ConfigPara
from config.config import ConfigDict


def train():
    use_gpu = ConfigPara.use_gpu
    # main(ConfigDict, use_cuda=use_gpu)
    spo_main(ConfigDict, use_cuda=use_gpu)


if __name__ == '__main__':
    # get_vocab(ConfigDict.get('train_data_path'), ConfigDict.get('test_data_path'))
    # get_p(ConfigDict.get('test_data_path'),ConfigDict.get('spo_test_data_path'))
    train()

