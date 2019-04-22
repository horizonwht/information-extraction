#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time   : 19-4-17 下午2:26
# @Author : wanghuiting
# @File   : config.py

import os

TEST_DEMO_PATH = os.path.join(os.getcwd(), "information-extraction/data/test_demo.json")
SO_TEST_DEMO_PATH = os.path.join(os.getcwd(), "information-extraction/data/test_demo.p")
TEST_DEMO_RES_PATH = os.path.join(os.getcwd(), "information-extraction/data/test_demo.res")
TEST_DEMO_RES_ZIP_PATH = os.path.join(os.getcwd(), "information-extraction/data/test_demo.res.zip")
DEV_DATA_PATH = os.path.join(os.getcwd(), "information-extraction/data/dev_data.json")
FINAL_P_MODEL_PATH = os.path.join(os.getcwd(), "information-extraction/model/p_model/final")
FINAL_SO_MODEL_PATH = os.path.join(os.getcwd(), "information-extraction/model/spo_model/final")
TEST_DEMO_SPO_PATH = os.path.join(os.getcwd(), "information-extraction/data/test_demo_spo.json")



word_idx_path = os.path.join(os.getcwd(), "information-extraction/dict/word_idx")
label_dict_path = os.path.join(os.getcwd(), "information-extraction/dict/p_eng")
so_label_dict_path = os.path.join(os.getcwd(), "information-extraction/dict/label_dict")
postag_dict_path = os.path.join(os.getcwd(), "information-extraction/dict/postag_dict")
train_data_path = os.path.join(os.getcwd(), "information-extraction/data/train_data.json")
test_data_path = os.path.join(os.getcwd(), "information-extraction/data/dev_data.json")
p_model_save_dir = os.path.join(os.getcwd(), "information-extraction/model/p_model")
spo_train_data_path = os.path.join(os.getcwd(), "information-extraction/data/train_data.p")
spo_test_data_path = os.path.join(os.getcwd(), "information-extraction/data/dev_data.p")
spo_model_save_dir = os.path.join(os.getcwd(), "information-extraction/model/spo_model")
emb_name = 'emb'
use_gpu = False
is_sparse = False
is_local = True
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
batch_size = 100
class_dim = 49

ConfigDict = {
    'word_idx_path': word_idx_path,
    'label_dict_path': label_dict_path,
    'so_label_dict_path': so_label_dict_path,
    'postag_dict_path': postag_dict_path,
    'train_data_path': train_data_path,
    'test_data_path': test_data_path,
    'p_model_save_dir': p_model_save_dir,
    'spo_train_data_path': spo_train_data_path,
    'spo_test_data_path': spo_test_data_path,
    'spo_model_save_dir': spo_model_save_dir,
    'emb_name':emb_name,
    'use_gpu':use_gpu,
    'is_sparse':is_sparse,
    'is_local':is_local,
    'word_emb_fixed':word_emb_fixed,
    'mix_hidden_lr':mix_hidden_lr,
    'cost_threshold':cost_threshold,
    'mark_dict_len':mark_dict_len,
    'word_dim':word_dim,
    'mark_dim':mark_dim,
    'postag_dim':postag_dim,
    'hidden_dim':hidden_dim,
    'depth':depth,
    'pass_num':pass_num,
    'batch_size':batch_size,
    'class_dim':class_dim,
}
