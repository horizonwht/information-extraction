#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time   : 19-4-17 下午4:43
# @Author : wanghuiting
# @File   : ConfigParameter.py

class ConfigPara:
    emb_name = 'emb'
    use_gpu = False
    is_sparse = False
    is_local = False
    word_emb_fixed = False
    mix_hidden_lr = 1e-3
    cost_threshold = 5
    mark_dict_len = 2
    word_dim = 128
    mark_dim = 5
    postag_dim = 20
    hidden_dim = 512
    depth = 8
    pass_num = 100
    batch_size = 1000
    class_dim = 49
