# Data Prepare Tools

![GitHub](https://img.shields.io/github/license/yourusername/amcadstudio) ![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/amcadstudio) [![GitHub issues](https://img.shields.io/github/issues/yourusername/amcadstudio)](https://github.com/yourusername/amcadstudio/issues)

**Author:** 邱鹏 (Qiu Peng)
**Content：** qiupeng@zhejianglab.com / qp0609@163.com

誓在2024/4/12前，从0-1训练一个1B大模型

## Installation
```bash
pip install -r requirements.txt
```

## Run script

1. 从jsonl文件中提取语料，并压缩成zstd格式
```bash
# arg1 输入文件夹
# arg2 输出文件夹
# arg3 并发数量
sh compress_jsonl_to_zstd.sh {your_workspace}/jsonl {your_workspace}/data 10
```
