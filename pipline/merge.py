# --*-- conding:utf-8 --*--
# @Time : 2024/5/14 13:54
# @Author : YWJ
# @Email : 52215901025@stu.ecnu.edu.cn
# @File : merge.py
# @Software : PyCharm
# @Description :  合并所有的json文件

import os
import json
import sys
from collections import defaultdict

def merge_json_files(input_dir, output_file):
    """
    合并所有的json文件
    :param input_dir: 输入目录
    :param output_file: 输出文件
    :return:
    """
    data = {}
    for file_name in os.listdir(input_dir):
        with open(os.path.join(input_dir, file_name), 'r', encoding='utf-8') as f:
            if "A12" in file_name:
                data["task1_A12"] = json.load(f)
            elif "A3" in file_name:
                data["task1_A3"] = json.load(f)
            elif "B1" in file_name:
                data["task1_B1"] = json.load(f)
            elif "NLI" in file_name:
                data["task2_NLI"] = json.load(f)
            else:
                print("wrong file name")
                exit(-1)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    input_dir = "/root/code/demo_project/A_test/predictions"
    output_file = "test_predictions.json"
    merge_json_files(input_dir, output_file)
