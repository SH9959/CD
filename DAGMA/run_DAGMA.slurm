#!/bin/bash
#SBATCH -J DAGMA                               # 作业名为 test
#SBATCH -o DAGMA-.out                        # 屏幕上的输出文件重定向到 test.out
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 01:01:59                           # 任务运行的最长时间为 1 小时
#SBATCH -w gpu01                              # 指定运行作业的节点是 gpu06，若不填写系统自动分配节点
##SBATCH --gres=gpu:tesla_p100-pcie-16gb:1
##SBATCH --cpus-per-task=8

# source ~/.bashrc

# 设置运行环境
conda activate hsong_CD

# 输入要执行的命令，例如 ./hello 或 python test.py 等
python DAGMA_test.py                    # 执行命令