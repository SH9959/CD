import os
import numpy as np
import pandas as pd

def check_path(path: str):  # 检查存在并新建
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        print("NotExist,Try to build a newone...")
        os.makedirs(directory)
        print("Built successfully")


def get_task_id(s: str):  # 从path中得到数字id

    import re
    import copy
    string = copy.deepcopy(s)
    match = re.search(r'dataset_(\d+)', string)
    # 检查是否匹配到字符串0
    if match:
        # 提取匹配到的数字部分并转换为整数
        number = int(match.group(1))
        print(number)  # 输出：1
        return number
    else:
        print("No match found in the string.")
        return None


def filter(prior_matrix: np.ndarray, wait_to_filter_matrix: np.ndarray):
    import copy
    m = copy.deepcopy(wait_to_filter_matrix)
    for r in range(0, len(prior_matrix)):
        for c in range(0, len(prior_matrix)):
            if prior_matrix[r][c] == 1:
                m[r][c] = 1
            if prior_matrix[r][c] == 0:
                m[r][c] = 0
    np.fill_diagonal(m, 0)
    return m


def dataset_to_Xcsv(path: str, win_size=300):
    import numpy as np
    import pandas as pd
    alarms = pd.read_csv(path)

    TIME_WIN_SIZE = win_size
    alarms = alarms.sort_values(by='start_timestamp')
    alarms['win_id'] = alarms['start_timestamp'].map(lambda elem: int(elem / TIME_WIN_SIZE))

    samples = alarms.groupby(['alarm_id', 'win_id'])['start_timestamp'].count().unstack('alarm_id')
    samples = samples.dropna(how='all').fillna(0)
    samples = samples.sort_index(axis=1)
    num = get_task_id(path)
    samples.to_csv(f'X_{num}.csv', index=False)
    print("ok")

def count_ones(matrix_path):
    matrix = np.load(matrix_path)
    count = 0  # 用于计数值为1的元素的个数
    # 遍历矩阵的每个元素
    for row in matrix:
        for element in row:
            if element == 1:
                count += 1
    return count

def draw(name="the name you want", path_list=[f"./dataset_{i}_graph_matrix.npy" for i in range(4, 7)]):
    import matplotlib.pyplot as plt
    m = []

    for path in path_list:
        a = np.load(path)
        m.append(a)
    # 创建一个4x4的子图布局
    plt.figure(figsize=(10, 10))

    # 绘制第一个子图（原始矩阵）
    for i in range(4, 7):

        plt.subplot(1, 3, i-3)
        plt.imshow(m[i-4], cmap='gray_r', vmin=m[i-4].min(), vmax=m[i-4].max())
        plt.colorbar()
        plt.title(f'{name}_d4')
        plt.xticks(np.arange(-0.5, m[i-4].shape[1] - 0.5, 1), np.arange(m[i-4].shape[1]))
        plt.yticks(np.arange(-0.5, m[i-4].shape[0] - 0.5, 1), np.arange(m[i-4].shape[0]))
        plt.grid(True, linestyle='--', alpha=0.7)
        ones_count = count_ones(m[i-4])
        print(f'{name}_d{i}: {ones_count}')
    #
    # # 绘制第二个子图（二进制矩阵）
    # plt.subplot(2, 2, 2)
    # plt.imshow(m[1], cmap='gray_r', vmin=m[1].min(), vmax=m[1].max())
    # plt.colorbar()
    # plt.title(f'{name}_d2')
    # plt.xticks(np.arange(-0.5, m[1].shape[1] - 0.5, 1), np.arange(m[1].shape[1]))
    # plt.yticks(np.arange(-0.5, m[1].shape[0] - 0.5, 1), np.arange(m[1].shape[0]))
    # plt.grid(True, linestyle='--', alpha=0.7)
    # ones_count = count_ones(m[1])
    # print(f'{name}_d2: {ones_count}')
    #
    # # 绘制第三个子图（黑色代表1，白色代表0）
    # plt.subplot(2, 2, 3)
    # plt.imshow(m[2], cmap='gray_r', vmin=m[2].min(), vmax=m[2].max())
    # plt.colorbar()
    # plt.title(f'{name}_d3')
    # plt.xticks(np.arange(-0.5, m[2].shape[1] - 0.5, 1), np.arange(m[2].shape[1]))
    # plt.yticks(np.arange(-0.5, m[2].shape[0] - 0.5, 1), np.arange(m[2].shape[0]))
    # plt.grid(True, linestyle='--', alpha=0.7)
    # ones_count = count_ones(m[2])
    # print(f'{name}_d3: {ones_count}')

    # # 绘制第四个子图（值越大越黑）
    # plt.subplot(2, 2, 4)
    # plt.imshow(m[3], cmap='gray_r', vmin=m[3].min(), vmax=m[3].max())
    # plt.colorbar()
    # plt.title(f'{name}_d4')
    # plt.xticks(np.arange(-0.5, m[3].shape[1] - 0.5, 1), np.arange(m[3].shape[1]))
    # plt.yticks(np.arange(-0.5, m[3].shape[0] - 0.5, 1), np.arange(m[3].shape[0]))
    # plt.grid(True, linestyle='--', alpha=0.7)
    # ones_count = count_ones(m[3])
    # print(f'{name}_d4: {ones_count}')

    # 调整子图之间的间距
    plt.tight_layout()
    plt.savefig(f'D:/1FromDesktop/Contests/Causal/结果.assets/{name}.png', dpi=300, bbox_inches='tight')

    # 显示所有子图
    plt.show()



def topology_filter(dt_idx, alarm_csv, topology_npy, result_npy, window_size=2):
    import os
    import numpy as np
    import pandas as pd
    import tqdm

    """
    生成 Legal_Edge矩阵 (包含所有的合法边，认为所有不在这个矩阵中的边都不合法)
    :param dt_idx: 数据集编号
    :param result_npy: 需要处理的最终结果 npy文件
    :param alarm_csv: 初始 Alarm数据 csv文件
    :param topology_npy: Topology先验 npy文件
    :param window_size: 时间窗大小
    :return: 没有返回值，会把 Legal_Edge矩阵保存为 LEGALEDGE/Legal_Edge_数据集编号_Win_时间窗大小.npy'
    """

    """  --------------------------- < 第一阶段：找出所有合法边 > ---------------------------  """

    """ 读取数据 """
    # 读取 需要处理的最终结果 npy文件
    DP_result = np.load(result_npy)
    # 读取初始 Alarm数据 csv文件
    Alarm = pd.read_csv(alarm_csv)
    # 读取 Topology先验 npy文件
    Topology = np.load(topology_npy)

    # 打印警报条目数量、警报类别数量、设备类别数量
    print("\n数据集id：", dt_idx)
    print("警报条目：", Alarm.shape[0])
    print("警报类别：", DP_result.shape[0])
    print("设备类别：", Topology.shape[0], "\n")

    ori_conut = 0
    for row in DP_result:
        for el in row:
            if el == 1:
                ori_count += 1
    print(f"数据集{dt_idx}初始边数：{ori_conut}")

    """ 筛选出所有的合法边 """
    if not os.path.exists('LEGALEDGE/Legal_Edge_' + str(dt_idx) + '_Win_' + str(window_size) + '.npy'):
        # 对 Alarm DataFrame按 start_timestamp 进行排序
        Alarm_sorted = Alarm.sort_values(by="start_timestamp")

        # 创建全 0的 Legal_Edge矩阵
        Legal_Edge = np.zeros((DP_result.shape[0], DP_result.shape[0]))

        # 使用一个长度为 window_size行的时间窗口扫描 Alarm_sorted DataFrame
        for start in tqdm.tqdm(range(0, Alarm_sorted.shape[0] - window_size + 1), desc="Scanning"):
            window_data = Alarm_sorted.iloc[start:start + window_size]

            # 创建一个临时的全 0的 Legal_Edge矩阵
            temp_Legal_Edge = np.zeros((DP_result.shape[0], DP_result.shape[0]))

            # 创建一个用于存储 window_size-相关设备 及 window_size-相关警报 的列表
            device_indices = []
            alarm_indices = []

            # 遍历 window_data 并找到所有 window_size-相关设备 及 window_size-相关警报
            for index, row in window_data.iterrows():
                device_indices.append(int(row['device_id']))
                alarm_indices.append(int(row['alarm_id']))

            for i in range(len(device_indices)):
                device = device_indices[i]
                for j in range(i, len(device_indices)):
                    if Topology[device][device_indices[j]] == 1:
                        temp_Legal_Edge[alarm_indices[i]][alarm_indices[j]] = 1

            # 合并 temp_Legal_Edge 和 Legal_Edge
            Legal_Edge = np.logical_or(Legal_Edge, temp_Legal_Edge).astype(int)

        # 保存 Legal_Edge矩阵
        np.save('LEGALEDGE/Legal_Edge_' + str(dt_idx) + '_Win_' + str(window_size) + '.npy', Legal_Edge)
    else:
        # 读取 LEGALEDGE/Legal_Edge.npy
        Legal_Edge = np.load('LEGALEDGE/Legal_Edge_' + str(dt_idx) + '_Win_' + str(window_size) + '.npy')

    # 打印 Legal_Edge矩阵
    print("Legal_Edge矩阵：\n", Legal_Edge)
    # 打印 Legal_Edge矩阵中 0的个数
    print("Legal_Edge矩阵中0的个数(非法边)：", np.sum(Legal_Edge == 0))
    # 检测 Legal_Edge矩阵是否对称
    print("Legal_Edge矩阵是否对称：", np.allclose(Legal_Edge, Legal_Edge.T), "\n")

    """  --------------------------- < 第二阶段：使用合法边过滤 > ---------------------------  """

    # 计数器记录改变的边的数量
    count = 0

    # 遍历 DP_result中的所有元素
    for i in range(DP_result.shape[0]):
        for j in range(DP_result.shape[1]):
            # 对于所有预测出来的边
            if DP_result[i][j] == 1:
                # 如果 Legal_Edge[i][j] == 0，将 DP_result[i][j]置为 0
                if Legal_Edge[i][j] == 0:
                    DP_result[i][j] = 0
                    count += 1

    print("改变的边的数量：", count)

    # 保存 DP_result矩阵
    # 如果不存在文件夹 PROCESSED/WIN_i，创建该文件夹
    if not os.path.exists('PROCESSED/WIN_' + str(window_size)):
        os.makedirs('PROCESSED/WIN_' + str(window_size))
    save_root = 'PROCESSED/WIN_' + str(window_size) + '/dataset_' + str(dt_idx) + '_topology_filter.npy'
    np.save(save_root, DP_result)

    print("\n保存路径：", save_root)

def rca_for_initial(rca_path:str, pri_path:str): # it may be worse
    '''
    :para pri_path is just used to get the size
    '''
    import pandas as pd
    rca = pd.read_csv(rca_path)
    pri = np.load(pri_path)
    print(rca)
    rca = pd.DataFrame(rca)
    tmp = np.zeros_like(pri)
    print(tmp,tmp.shape[0], tmp.shape[1])
    for index, row in rca.iterrows():
        simplified_snapshot = row['simplified_snapshot']
        simplified_root_cause = row['simplified_root_cause']
        print(type(simplified_snapshot), type(simplified_root_cause))
        # 从字符串转换为元组
        str_tuple = simplified_snapshot
        tuple_from_str = eval(str_tuple)
        # 如果需要将元组中的元素转换为整数
        int_list = list(int(x) for x in tuple_from_str)
        # 将两两之间的值置为1
        for i in int_list:
            for j in int_list:
                if i != j:
                    tmp[i][j] = 1

    for index, row in rca.iterrows():   
        simplified_snapshot = row['simplified_snapshot']
        simplified_root_cause = row['simplified_root_cause']
        print(type(simplified_snapshot), type(simplified_root_cause))
        # 从字符串转换为元组
        str_tuple = simplified_snapshot
        tuple_from_str = eval(str_tuple)
        # 如果需要将元组中的元素转换为整数
        int_list = list(int(x) for x in tuple_from_str)
        for i in int_list:
            tmp[i][simplified_root_cause] = 0
        print(tmp,tmp.shape[0], tmp.shape[1])
        cc = 0
        for i in tmp:
            for j in i:
                if j == 1:
                    cc += 1
        print(tmp.shape[0]*tmp.shape[1]-cc)

    print(tmp,tmp.shape[0], tmp.shape[1])
    cc = 0
    for i in tmp:
        for j in i:
            if j == 1:
                cc += 1
    print(tmp.shape[0]*tmp.shape[1]-cc)
    #np.save("./init_matrix_phase2/rca_for_initial/data_5_.npy", tmp)

def rca_for_filter(rca_path:str, pri_path:str): # 可以并上初始化矩阵，也可以过滤成果矩阵（有待验证）

    rca = pd.read_csv(rca_path)
    pri = np.load(pri_path)
    print(rca)
    rca = pd.DataFrame(rca)
    tmp = np.zeros_like(pri)
    print(tmp,tmp.shape[0], tmp.shape[1])
    dic = {}
    for index, row in rca.iterrows():
        simplified_snapshot = row['simplified_snapshot']
        simplified_root_cause = row['simplified_root_cause']
        print(type(simplified_snapshot), type(simplified_root_cause))
        # 从字符串转换为元组
        str_tuple = simplified_snapshot
        tuple_from_str = eval(str_tuple)
        # 如果需要将元组中的元素转换为整数
        int_list = set(int(x) for x in tuple_from_str)
        for item in int_list:
            current_simplified_root_cause = simplified_root_cause
            # 如果当前simplified_root_cause不在字典中，初始化一个空集合
            if current_simplified_root_cause not in dic:
                dic[current_simplified_root_cause] = set()
            # 将item添加到对应的集合中
            dic[current_simplified_root_cause].add(item)

    print(dic)
    dic2 = {}
    for key, v in dic.items():
        print(key, v)
        for i in range(0, tmp.shape[0]):
            print("i", type(i))
            print("v", type(v))
            if i not in v:
                 if key not in dic2:
                     dic2[key] = set()
                 dic2[key].add(i)
    for k, v in dic2.items():
        for causer_candidate in v:
            if causer_candidate in dic2:
                tmp[causer_candidate][int(k)] = 1
    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            if tmp[i][j] == 1:
                print("tmp")
                print(i, j)
            if pri[i][j] == 1:
                print("pri")
                print(i,j)
    p = "./RESULTS/rca_prior/rca_filterr_6.npy"
    check_path(p)
    np.save(p, tmp)
    print("saved!!!")
    print(dic2)

def rca_intersection(rca_path:str) -> dict:  # return intersection
    rca = pd.read_csv(rca_path)
    print(rca)
    rca = pd.DataFrame(rca)
    dic = {}
    for index, row in rca.iterrows():
        simplified_snapshot = row['simplified_snapshot']
        simplified_root_cause = row['simplified_root_cause']
        print(type(simplified_snapshot), type(simplified_root_cause))
        # 从字符串转换为元组
        str_tuple = simplified_snapshot
        tuple_from_str = eval(str_tuple)
        # 如果需要将元组中的元素转换为整数
        int_set = set(int(x) for x in tuple_from_str)
        if simplified_root_cause not in dic:
            dic[simplified_root_cause] = int_set
        # 将item添加到对应的集合中
            print(simplified_root_cause)
        dic[simplified_root_cause] = dic[simplified_root_cause].intersection(int_set)

    print(dic)
    return dic

def check_result_with_strong_sons_from_rca(rca_path:str, result:str):

    dic = rca_intersection(rca_path)
    resul = np.load(result)
    tmp = np.zeros_like(resul)
    for k, v in dic.items():
        for i in v:
            if resul[int(k)][i] == 1 and int(k) != i:
                print(int(k), i)
            else:
                print(f" create {i}-> {v} - {i}")
                for ii in v:
                    if ii != i:
                        tmp[i][ii] = 1
    num = get_task_id(result)
    savepath=f"./RESULTS/rca_add_others_to_v/dataset_{num}_rca_est.npy"
    check_path(savepath)
    np.save(savepath, tmp)
    print(f"saved in {savepath}")
    return tmp

def inter_union(m1:str, m2:str, savepath:str, model="union"):
    '''
    :para: m1 
    '''
    a = np.load(m1)
    b = np.load(m2)
    if model == "union":
        result = np.logical_or(a, b)
    elif model == "intersect":
        result = np.logical_and(a, b)
    else:
        print("out of range")
    check_path(savepath)
    np.save(savepath, result)
    return result

if __name__ == "__main__":
    data_id = 6
    rca = f"./datasets_phase2/dataset_{data_id}/rca_prior.csv"
    pri = f"./datasets_phase2/dataset_{data_id}/causal_prior.npy"
    result = f"./RESULTS/BEST-1025/dataset_{data_id}_graph_matrix.npy"
    #print(type(rca))
    #rca_for_initial(rca, pri)
    #rca_for_filter(rca, pri)
    #rca_intersection(rca_path = rca)
    check_result_with_strong_sons_from_rca(rca_path=rca, result=result)
    #     # topology_filter(6, './dataset_phase2/alarm.csv', 'ALARM_PRIOR/1/topology.npy', 'DAGMA50/dataset_1_graph_matrix.npy', 2)
    # before = "./Notears_result/dataset_5_graph_matrix.npy"
    # now1 = "./Notears_result_win1000/dataset_5_graph_matrix.npy"
    # now2 = "./Notears_result_win2000/dataset_5_graph_matrix.npy"
    # count_ones(before)
    # count_ones(now1)
    # count_ones(now2)
    # para = {
    #     'dt_idx': 5, 
    #     'alarm_csv': "../datasets_phase2/dataset_5/alarm.csv", 
    #     'topology_npy': "../datasets_phase2/dataset_5/topology.npy", 
    #     'result_npy': "./Notears_result_win1000/dataset_5_graph_matrix.npy", 
    #     'window_size': 2
    # }

    # topology_filter(dt_idx=para['dt_idx'], alarm_csv=para['alarm_csv'], topology_npy=para['topology_npy'] , result_npy=para['result_npy'] ,window_size=para['window_size'])
  