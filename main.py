#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time   : 19-4-17 下午2:25
# @Author : wanghuiting
# @File   : main.py

from lib.get_vocab import get_vocab
from lib.get_spo_train import get_p
from bin.p_classification.p_train import main
from bin.so_labeling.spo_train import spo_main
from config.ConfigParameter import ConfigPara
from config.config import ConfigDict, FINAL_P_MODEL_PATH, FINAL_SO_MODEL_PATH, TEST_DEMO_PATH, SO_TEST_DEMO_PATH, \
    TEST_DEMO_RES_PATH, TEST_DEMO_RES_ZIP_PATH,TEST_DEMO_SPO_PATH
from bin.p_classification.p_infer import p_infer_main
from bin.so_labeling.spo_infer import spo_main
from bin.evaluation.calc_pr import calc_pr
import json


def train():
    use_gpu = ConfigPara.use_gpu
    main(ConfigDict, use_cuda=use_gpu)
    # spo_main(ConfigDict, use_cuda=use_gpu)


def test():
    p_infer_main(ConfigDict, FINAL_P_MODEL_PATH, TEST_DEMO_PATH, SO_TEST_DEMO_PATH)
    spo_main(ConfigDict, FINAL_SO_MODEL_PATH, SO_TEST_DEMO_PATH, TEST_DEMO_RES_PATH)


def evaluat():
    ret_info = calc_pr(TEST_DEMO_RES_ZIP_PATH, '', '', TEST_DEMO_SPO_PATH)
    print(json.dumps(ret_info))


if __name__ == '__main__':
    # get_vocab(ConfigDict.get('train_data_path'), ConfigDict.get('test_data_path'))
    # get_p(ConfigDict.get('train_data_path'))
    # train()
    # test()
    evaluat()

a = [
    '{"postag": [{"pos": "w", "word": "《"}, {"pos": "nw", "word": "我们的少年时代"}, {"pos": "w", "word": "》"}, {"pos": "nr", "word": "程砚秋"}, {"pos": "p", "word": "在"}, {"pos": "r", "word": "该剧"}, {"pos": "f", "word": "中"}, {"pos": "v", "word": "饰演"}, {"pos": "n", "word": "富家千金"}, {"pos": "w", "word": "，"}, {"pos": "v", "word": "陶西"}, {"pos": "n", "word": "前女友"}],'
    ' "text": "《我们的少年时代》程砚秋在该剧中饰演富家千金，陶西前女友", '
    '"spo_list": [{"object": "程砚秋", "object_type": "人物", "predicate": "主演", "subject": "我们的少年时代", "subject_type": "影视作品"}]}',
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [23, 7, 23, 4, 19, 18, 1, 9, 0, 23, 9, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
train_data = {"postag":
         [{"word": "横沥镇", "pos": "ns"}, {"word": "属", "pos": "v"}, {"word": "亚热带季风气候", "pos": "nz"},
          {"word": "地区", "pos": "n"}, {"word": "，", "pos": "w"}, {"word": "季风", "pos": "n"}, {"word": "明显", "pos": "a"},
          {"word": "，", "pos": "w"}, {"word": "气候", "pos": "n"}, {"word": "温和", "pos": "a"}],
     "text": "横沥镇属亚热带季风气候地区，季风明显，气候温和",
     "spo_list": [
         {"predicate": "气候", "object_type": "气候", "subject_type": "行政区", "object": "亚热带季风气候", "subject": "横沥镇"}]}

test_input = {
    "postag": [{"word": "华帝燃气热水器华帝燃具股份有限公司", "pos": "nt"}, {"word": "成立", "pos": "v"}, {"word": "于", "pos": "p"},
               {"word": "2001年11月28日", "pos": "t"}, {"word": "，", "pos": "w"}, {"word": "主要", "pos": "ad"},
               {"word": "从事", "pos": "v"}, {"word": "生产", "pos": "vn"}, {"word": "和", "pos": "c"},
               {"word": "销售", "pos": "vn"}, {"word": "燃气", "pos": "n"}, {"word": "用具", "pos": "n"},
               {"word": "、", "pos": "w"}, {"word": "厨房", "pos": "n"}, {"word": "用具", "pos": "n"},
               {"word": "、", "pos": "w"}, {"word": "家用电器", "pos": "n"}, {"word": "及", "pos": "c"},
               {"word": "企业", "pos": "n"}, {"word": "自有", "pos": "v"}, {"word": "资产", "pos": "n"},
               {"word": "、", "pos": "w"}, {"word": "进出口", "pos": "vn"}, {"word": "经营", "pos": "vn"},
               {"word": "业务", "pos": "n"}, {"word": "，", "pos": "w"}, {"word": "华帝", "pos": "nt"},
               {"word": "产品", "pos": "n"}, {"word": "已", "pos": "d"}, {"word": "形成", "pos": "v"},
               {"word": "燃气", "pos": "n"}, {"word": "灶具", "pos": "n"}, {"word": "、", "pos": "w"},
               {"word": "热水器", "pos": "n"}, {"word": "（", "pos": "w"}, {"word": "电", "pos": "n"},
               {"word": "热水器", "pos": "n"}, {"word": "、", "pos": "w"}, {"word": "燃气", "pos": "vn"},
               {"word": "热水器", "pos": "n"}, {"word": "和", "pos": "c"}, {"word": "太阳能", "pos": "n"},
               {"word": "热水器", "pos": "n"}, {"word": "）", "pos": "w"}, {"word": "、", "pos": "w"},
               {"word": "抽油烟机", "pos": "n"}, {"word": "、", "pos": "w"}, {"word": "消毒柜", "pos": "n"},
               {"word": "、", "pos": "w"}, {"word": "橱柜", "pos": "n"}, {"word": "等", "pos": "u"},
               {"word": "系列", "pos": "n"}, {"word": "产品", "pos": "n"}, {"word": "为主", "pos": "v"},
               {"word": "的", "pos": "u"}, {"word": "500多个", "pos": "m"}, {"word": "品种", "pos": "n"},
               {"word": "，", "pos": "w"}, {"word": "燃气", "pos": "n"}, {"word": "灶具", "pos": "n"},
               {"word": "连续", "pos": "a"}, {"word": "十一年", "pos": "m"}, {"word": "中国", "pos": "ns"},
               {"word": "产", "pos": "v"}, {"word": "销量", "pos": "n"}, {"word": "，", "pos": "w"},
               {"word": "燃气", "pos": "n"}, {"word": "热水器", "pos": "n"}, {"word": "、", "pos": "w"},
               {"word": "抽油烟机", "pos": "n"}, {"word": "分别", "pos": "d"}, {"word": "进入", "pos": "v"},
               {"word": "全国", "pos": "n"}, {"word": "行业", "pos": "n"}, {"word": "三强", "pos": "n"}],
    "text": "华帝燃气热水器华帝燃具股份有限公司成立于2001年11月28日，主要从事生产和销售燃气用具、厨房用具、家用电器及企业自有资产、进出口经营业务，华帝产品已形成燃气灶具、热水器（电热水器、燃气热水器和太阳能热水器）、抽油烟机、消毒柜、橱柜等系列产品为主的500多个品种，燃气灶具连续十一年中国产销量，燃气热水器、抽油烟机分别进入全国行业三强"}

test_result = {
    "spo_list": [
        {"object": "1958年", "predicate": "出生日期", "subject": "刘一鸣", "object_type": "Date", "subject_type": "人物"},
        {"object": "江苏省苏州市相城区阳澄湖", "predicate": "出生地", "subject": "刘一鸣", "object_type": "地点", "subject_type": "人物"}],
    "text": "刘一鸣，男，生于1958年，江苏省苏州市相城区阳澄湖人，自小随祖父、父亲学习雕刻"}

