# -*- coding: utf-8 -*-
########################################################
# Copyright (c) 2019, Baidu Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# imitations under the License.
########################################################
"""
This module to generate training data for training a so-labeling model
"""

import codecs
import json



def get_p(input_file,output_file):
    """
    Generate training data for so labeling model
    """
    result = []
    with codecs.open(input_file, 'r', 'utf-8') as fr:
        for line in fr:
            try:
                dic = json.loads(line.strip())
                # print(dic)
            except:
                continue
            spo_list = dic['spo_list']
            # print(spo_list)
            p_list = [item['predicate'] for item in spo_list]
            for p in p_list:
                result.append([json.dumps(dic),p])
    # print(result)

    with codecs.open(output_file, 'w', 'utf-8') as wd:
        for line in result:
            wd.writelines('\t'.join(line) + '\n')

