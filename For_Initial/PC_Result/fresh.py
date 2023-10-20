import Utils
import numpy as np
import os
pri = [f'D:/1FromDesktop/Contests/Causal/code/CD/datasets_phase2/dataset_{i}/causal_prior.npy' for i in range(4,7)]
# 指定目录路径
directory_path = "D:/1FromDesktop/Contests/Causal/code/CD/For_Initial/PC_Result"  # 将路径替换为你要遍历的目录路径

# 使用列表推导式获取目录中所有后缀为.npy的文件
npy_files = [filename for filename in os.listdir(directory_path) if filename.endswith(".npy")]

# 打印所有后缀为.npy的文件列表
paths = []
for npy_file in npy_files:
    paths.append(npy_file)

    print(npy_file)

for path in paths:
    n = Utils.get_task_id(path)
    prio = np.load(pri[n-4])
    a = np.load(path)
    a = Utils.filter(wait_to_filter_matrix=a, prior_matrix=prio)
    np.save(path, a)
    print("ok")