import numpy as np
import pandas as pd
import sys
import os
sys.path.append(r"D:\1FromDesktop\Contests\Causal\code\trustworthyAI\gcastle")
from castle.metrics import MetricsDAG
from castle.common.plot_dag import GraphDAG

data1 = np.load("./weighted_form/dataset_1_graph_matrix.npy")
data2 = np.load("./weighted_form/dataset_2_graph_matrix.npy")
data3 = np.load("./weighted_form/dataset_3_graph_matrix.npy")
data4 = np.load("./weighted_form/dataset_4_graph_matrix.npy")
# GraphDAG(data1)
# GraphDAG(data2)
# GraphDAG(data3)
# GraphDAG(data4)

D = [data1, data2, data3, data4]

for a in range(5, 36, 5):  # 0.05-0.35
    threshold = a / 100
    k = 0

    # 构建目标文件夹路径
    folder_path = f"../SAM_submission/bin_form/thres_{threshold}"

    for data in D:
        k += 1
        # 设置一个阈值
        binary_W_est = (data > threshold).astype(int)



        # 检查目标文件夹是否存在，如果不存在则创建它
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 构建文件路径
        file_path = f"{folder_path}/dataset_{k}_graph_matrix.npy"

        # 保存二进制图矩阵到文件
        np.save(file_path, binary_W_est)

print("ok")
