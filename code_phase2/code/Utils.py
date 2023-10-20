import os
import numpy as np


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
    # 检查是否匹配到字符串
    if match:
        # 提取匹配到的数字部分并转换为整数
        number = int(match.group(1))
        print(number)  # 输出：1
        return number
    else:
        print("No match found in the string.")
        return None


def filter(prior_matrix:np.ndarray, wait_to_filter_matrix:np.ndarray):
    import copy
    m = copy.deepcopy(wait_to_filter_matrix)
    if prior_matrix is not None:
        for r in range(0, len(wait_to_filter_matrix)):
            for c in range(0, len(wait_to_filter_matrix)):
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
    samples.to_csv(f'./X_{num}.csv', index=False)
    print("ok")