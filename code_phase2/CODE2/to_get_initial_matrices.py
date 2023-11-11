    
import json
import Utils
import numpy as np
import pandas as pd
import time, os
from copy import deepcopy
# Load the JSON file
with open('datapath.json', 'r') as json_file:
    params = json.load(json_file)

# Access the parameters
datasets = params['datasets']
pri = params['pri']
rca = params['rca']
topo = params['topo']
init_paths = params['init_paths']
SAVEPATH = params['SAVEPATH']

def DAGMA(data_id:int):  # 4, 5, 6
    print("DAGMA is running")
    TIME_WIN_SIZE = 300
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


    # 读取 CSV 文件
    # 3
    global pri
    global datasets
    alarms = pd.read_csv(datasets[data_id - 4])
    pri_path = np.load(pri[data_id - 4])
    

    alarms = alarms.sort_values(by='start_timestamp')
    alarms['win_id'] = alarms['start_timestamp'].map(lambda elem:int(elem/TIME_WIN_SIZE))

    samples=alarms.groupby(['alarm_id', 'win_id'])['start_timestamp'].count().unstack('alarm_id')
    samples = samples.dropna(how='all').fillna(0)
    samples = samples.sort_index(axis=1)
    # 4
    # samples.to_csv(f'X{i}.csv', index=False)
    # # print("ok")
    # data = pd.read_csv(f'X{i}.csv')

    data = deepcopy(samples)
    print("DEBUG data.type", type(data))
    # 将数据转换为 NumPy 数组
    X_numpy = data.to_numpy()
    print("X_numpy.shape:", X_numpy.shape)
    A=time.time()
    model = DagmaLinear(loss_type='l2')  # create a linear model with least squares loss

    # 将 self.X 转换为浮点数类型
    X_numpy = X_numpy.astype(np.float64)

    W_est = model.fit(X_numpy, lambda1=0.02)  # fit the model with L1 reg. (coeff. 0.02)
  
    print("W_est", W_est)
    B=time.time()
    print(f"time: {B-A}s")
    # np.save("DAGMA_est_graph.npy", W_est)
    # print("B_True\n", B_true)
    # acc = utils.count_accuracy(B_true, W_est != 0) # compute metrics of estimated adjacency matrix W_est with ground-truth
    # print(acc)

    threshold = 0.05
    binary_W_est = (W_est > threshold).astype(int)
    print("Binary W_est:\n", binary_W_est)
    # p = f"./DAGMA_phase2_win{TIME_WIN_SIZE}/thres_{threshold}/dataset_{i}_graph_matrix.npy"
    # p = f"./init_matrixes/init_matrix_from_DAGMA{TIME_WIN_SIZE}_"
    # Utils.check_path(p)
    binary_W_est = Utils.filter(prior_matrix=pri, wait_to_filter_matrix=binary_W_est)
    
    # np.save(p, binary_W_est)
    # print("saved!")
    return binary_W_est


def origin_data_to_X(data_id:int):
    """
    官方给的原始数据转换得到X.csv
    """
    global datasets
    alarms = pd.read_csv(datasets[data_id - 4])
    TIME_WIN_SIZE  = 300
    #Use a time sliding window(300 seconds) to generate IID samples.
    alarms = alarms.sort_values(by='start_timestamp')
    alarms['win_id'] = alarms['start_timestamp'].map(lambda elem:int(elem/TIME_WIN_SIZE))

    samples=alarms.groupby(['alarm_id','win_id'])['start_timestamp'].count().unstack('alarm_id')
    samples = samples.dropna(how='all').fillna(0)
    samples = samples.sort_index(axis=1)
    print("samples.shape:", samples.shape)
    #np.save("X.csv", samples)  #暂时不保存了
    return samples

def PC(data_id:int):
    from trustworthyAI.gcastle.castle.algorithms import PC
    from trustworthyAI.gcastle.castle.common.priori_knowledge import PrioriKnowledge
    """
    PC方法需要git clone trustworthyAI库,记得改绝对路径,
    dataset1: (2017,39) 8s
    dataset2: (2016,49) 37s
    dataset3: (2017,31) 3s
    dataset4: (6582,30) 3000s
    """
    print(" ")
    print("PC is running")
    t1 = time.time()

    # causal_prior
    global pri
    global datasets
    alarms = pd.read_csv(datasets[data_id - 4])
    causal_prior = np.load(pri[data_id - 4])
    print(f"shape of alarm data: {alarms.shape}")
    print(f"shape of causal prior matrix: {causal_prior.shape}")
    # Notes: topology.npy and rca_prior.csv are not used in this script.

    samples = origin_data_to_X(data_id)

    # create the prior knowledge object for the PC algorithm 
    prior_knowledge = PrioriKnowledge(causal_prior.shape[0])
    for i, j in zip(*np.where(causal_prior == 1)):
        prior_knowledge.add_required_edge(i, j)

    for i, j in zip(*np.where(causal_prior == 0)):
        prior_knowledge.add_forbidden_edge(i, j)

    pc = PC(priori_knowledge=prior_knowledge)
    pc.learn(samples)
    graph_matrix = np.array(pc.causal_matrix)
    print("graph_matrix.shape:", graph_matrix.shape)
    t2 = time.time()

    print(f"time: {t2-t1}s")
    return graph_matrix

import os
import numpy as np
import pandas as pd


def load_data(
        data_id: int,
        data_basepath: str
):
    # Read historic alarm data
    data_path = os.path.join(data_basepath, str("dataset_" + str(data_id)))
    alarm_data = pd.read_csv(os.path.join(data_path, "alarm.csv"), encoding='utf')

    return alarm_data


def learn_effect(
    data_id: int,
    win_size=12,
    clip_stats=True,
    data_basepath="datasets_phase2/",
    sava_path="effect/",
):
    # alarm_data, columns: alarm_id, device_id, start_timestamp, end_timestamp
    alarm_data = load_data(data_id, data_basepath=data_basepath)

    event_ids = sorted(alarm_data["alarm_id"].unique())
    num_events = len(event_ids)
    # print("event_ids:", event_ids, "num_events:", num_events)

    # Get duration
    alarm_data["duration"] = alarm_data["end_timestamp"] - alarm_data["start_timestamp"]
    data_times = alarm_data[["alarm_id", "start_timestamp", "end_timestamp", "duration"]].values

    # alarm_id onehot
    alarm_onehot_df = pd.get_dummies(alarm_data["alarm_id"], prefix="e", columns=["alarm_id"])
    # print(alarm_onehot_df[:10])

    delta_index = win_size
    data_onehot = alarm_onehot_df.values
    # print(data_onehot.shape)
    # print(data_onehot[:10])

    # apply sliding window, [N, num_events] --> [N, num_events, win_size]
    data_view = np.lib.stride_tricks.sliding_window_view(data_onehot, delta_index, axis=0)
    print("data_view:", data_view.shape)
    # print(data_view[1])

    data_pos = np.copy(data_view)
    data_pos[data_view == 0] = win_size
    # print(data_pos[1])
    happen_times = np.argmin(data_pos, axis=-1)
    # print(happen_times[1])

    data_counts = np.sum(data_view, axis=-1)
    happen_times[data_counts == 0] = 13
    event_happen_times = happen_times.T
    # print(happen_times[0])
    # print(happen_times[1])

    # [N, 3] --> [N, 4, win_size]
    times_view = np.lib.stride_tricks.sliding_window_view(data_times, delta_index, axis=0)
    times_alarms = times_view[:, 0, 1:]
    times_deltas = times_view[:, 1:, 1:] - times_view[:, 1:, 0:1]
    # times_deltas = np.log(times_deltas)
    print("times_view.shape: {}, times_alarms.shape: {}, times_deltas.shape: {}".format(
        times_view.shape, times_alarms.shape, times_deltas.shape)
    )

    # sum over the sliding window, [N, num_events, win_size-1] --> [N, num_events]
    data_final = np.sum(data_view[:, :, 1:], axis=-1)

    if clip_stats:
        data_final_clipped = np.clip(data_final, a_min=0, a_max=1)
    else:
        data_final_clipped = data_final

    effect_matrix_pos = np.zeros((num_events, num_events), dtype=np.float32)
    effect_matrix_neg = np.zeros((num_events, num_events), dtype=np.float32)
    avg_effect_times_pos = np.zeros((num_events, num_events), dtype=np.float32)
    avg_effect_times_neg = np.zeros((num_events, num_events), dtype=np.float32)
    std_effect_times_pos = np.zeros((num_events, num_events), dtype=np.float32)
    std_effect_times_neg = np.zeros((num_events, num_events), dtype=np.float32)

    for i in range(num_events):
        for j in range(num_events):
            if i == j:
                continue
            # samples started with i
            selected_rows = event_happen_times[i] == 0
            selected_data = data_final_clipped[selected_rows]
            effect_matrix_pos[i, j] = np.mean(selected_data[:, j])
            # samples started with not i
            remained_rows = event_happen_times[i] != 0
            remained_data = data_final_clipped[remained_rows]
            effect_matrix_neg[i, j] = np.mean(remained_data[:, j])

            # time stats
            selected_time_deltas = times_deltas[selected_rows]
            window_mask = (times_alarms[selected_rows] == j).astype(np.float32)
            time_stats_sum = np.sum(selected_time_deltas * window_mask[:, np.newaxis, :], axis=-1)
            # print(selected_time_deltas.shape, time_stats_sum.shape)
            effected_times = time_stats_sum[time_stats_sum[:, 0] > 0, 0]
            avg_effect_times_pos[i, j] = np.mean(effected_times)
            std_effect_times_pos[i, j] = np.std(effected_times)

            # time stats
            selected_time_deltas = times_deltas[remained_rows]
            window_mask = (times_alarms[remained_rows] == j).astype(np.float32)
            time_stats_sum = np.sum(selected_time_deltas * window_mask[:, np.newaxis, :], axis=-1)
            effected_times = time_stats_sum[time_stats_sum[:, 0] > 0, 0]
            avg_effect_times_neg[i, j] = np.mean(effected_times)
            std_effect_times_neg[i, j] = np.std(effected_times)

    effect_matrix = effect_matrix_pos - effect_matrix_neg

    # np.save(sava_path + "effect_matrix_" + str(data_id) + ".npy", effect_matrix)

    return effect_matrix

def effect_genera(
    data_id: int,
    threshold = 0.2,
    win_size=16,
    data_basepath="datasets_phase2/",
    sava_path="effect/",
):

    effect_matrix = learn_effect(
        data_id,
        win_size=win_size,
        clip_stats=False,
        data_basepath=data_basepath
    )

    """阈值过滤"""
    processed_matrix = np.copy(effect_matrix)
    # 替换大于等于阈值的元素为1
    processed_matrix[processed_matrix >= threshold] = 1
    # 替换小于等于阈值的相反数的元素为-1
    processed_matrix[processed_matrix <= -threshold] = -1
    # 替换其余的元素为0
    processed_matrix[(processed_matrix < threshold) & (processed_matrix > -threshold)] = 0
    # 计算非零元素的数量
    non_zero_count = np.count_nonzero(processed_matrix)

    # print("处理后的矩阵:")
    # print(processed_matrix)
    print(f"非零元素的数量: {non_zero_count}\n")

    """转换为有向图"""
    N = processed_matrix.shape[0]
    effect_dag = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if processed_matrix[i][j] == 1:
                effect_dag[i][j] = 1
            elif processed_matrix[i][j] == -1:
                effect_dag[j][i] = 1

    # print("\n有向图:")
    # print(effect_dag)

    # 保存为npy文件
    # np.save(sava_path + 'processed_dag_' + str(data_id) + '.npy', processed_matrix)

    return processed_matrix

def cf_for_dts(data_id):
    win_size = 16
    threshold = 0.2

    processed_matrix = effect_genera(data_id=data_id,
                                     win_size=win_size,
                                     threshold=threshold)

    return processed_matrix


if __name__ == "__main__":
    for i in range(4,7):
        # get DAGMA results
        dagma_result = DAGMA(i)
        path =f"./_init_matrices/dagma/init_{i}_graph.npy"
        Utils.check_path(path)
        np.save(path, dagma_result)

        # get PC results
        PC_result = PC(i)
        path =f"./_init_matrices/PC/init_{i}_graph.npy"
        Utils.check_path(path)
        np.save(path, dagma_result)

        # get causal effect results
        causal_effect_matrix = cf_for_dts(i)
        path =f"./_init_matrices/causal_effect/init_{i}_graph.npy"
        Utils.check_path(path)
        np.save(path, causal_effect_matrix)



