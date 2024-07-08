#!/bin/bash

# 定义需要的codebook_size值
codebook_sizes=(32 64 128 256 320 512 640 768 896 1024 1280 1520)

# 文件路径
file_path="/datadrive/yingwei/LangSplat/gaussian_renderer/__init__.py"
output_file="output.info"



# 遍历每一个codebook_size值
for size in "${codebook_sizes[@]}"; do
    # 使用sed命令修改文件的第109行
    sed -i "109s/codebook_size=[0-9]*/codebook_size=${size}/" $file_path

    # 运行process.sh脚本
    ./process.sh

    # 提取并附加指定的输出信息
    grep "INFO:teatime:iou chosen:" <(./process.sh) >> $output_file
done

echo "所有操作完成，输出已保存到 $output_file"
