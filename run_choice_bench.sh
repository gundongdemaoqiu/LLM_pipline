#!/bin/bash

# 设置环境变量
export HF_HOME=/mnt/vepfs/fs_users/lkn/huggingface
export http_proxy=100.68.173.80:3128
export https_proxy=100.68.173.80:3128

# 运行 Python 脚本并将输出重定向到日志文件
nohup python pipline/choice_bench.py > output.log 2>&1 &
