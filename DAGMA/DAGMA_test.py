import time

import torch
from dagma import utils
from dagma.linear import DagmaLinear
from dagma.nonlinear import DagmaMLP, DagmaNonlinear

utils.set_random_seed(1)
# Create an Erdos-Renyi DAG of 20 nodes and 20 edges in expectation with Gaussian noise
# number of samples n = 500
n, d, s0 = 500, 20, 20
graph_type, sem_type = 'ER', 'gauss'

B_true = utils.simulate_dag(d, s0, graph_type)
W_true = utils.simulate_parameter(B_true)
X = utils.simulate_linear_sem(W_true, n, sem_type)
print(X.shape)

import Utils
import numpy as np
import pandas as pd
import time, os
# 读取 CSV 文件
# 3
for i in range(4,7):

    alarms = pd.read_csv(f'./datasets_phase2/dataset_{i}/alarm.csv')
    pri_path = f"./datasets_phase2/dataset_{i}/causal_prior.npy"
    

    TIME_WIN_SIZE = 300
    alarms = alarms.sort_values(by='start_timestamp')
    alarms['win_id'] = alarms['start_timestamp'].map(lambda elem:int(elem/TIME_WIN_SIZE))

    samples=alarms.groupby(['alarm_id', 'win_id'])['start_timestamp'].count().unstack('alarm_id')
    samples = samples.dropna(how='all').fillna(0)
    samples = samples.sort_index(axis=1)
    # 4
    samples.to_csv(f'X{i}.csv', index=False)
    # print("ok")
    data = pd.read_csv(f'X{i}.csv')

    # 将数据转换为 NumPy 数组
    X_numpy = data.to_numpy()
    print("X_numpy.shape:", X_numpy.shape)
    A=time.time()
    model = DagmaLinear(loss_type='l2')  # create a linear model with least squares loss

    # 将 self.X 转换为浮点数类型
    X_numpy = X_numpy.astype(np.float64)


    W_est = model.fit(X_numpy, lambda1=0.02)  # fit the model with L1 reg. (coeff. 0.02)

    if not os.path.exists(pri_path):
        for i in range(len(W_est)):
            for j in range(len(W_est)):
                pri[i][j] = -1
    else:
        pri = np.load(pri_path)
        
    print("W_est", W_est)
    B=time.time()
    print(f"time: {B-A}s")
    # np.save("DAGMA_est_graph.npy", W_est)
    # print("B_True\n", B_true)
    # acc = utils.count_accuracy(B_true, W_est != 0) # compute metrics of estimated adjacency matrix W_est with ground-truth
    # print(acc)

    for a in range(5,45):
        threshold = a/100

        binary_W_est = (W_est > threshold).astype(int)
        print("Binary W_est:\n", binary_W_est)
        p = f"./DAGMA_phase2/thres_{threshold}/dataset_{i}_graph_matrix.npy"
        Utils.check_path(p)
        binary_W_est = Utils.filter(prior_matrix=pri, wait_to_filter_matrix=binary_W_est)

        np.save(p, binary_W_est)
        print("saved!")
