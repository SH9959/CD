import numpy as np
import pandas as pd

# 假设 true_graph 是一个 NumPy 数组，已经加载好了
# true_graph = np.load(r'D:/1FromDesktop/Contests/Causal/code/trustworthyAI/competition/NeurIPS2023/sample/true_graph.npy')

# 将 NumPy 数组转换为 Pandas DataFrame

true_graph = np.load(r'D:/1FromDesktop/Contests/Causal/code/trustworthyAI/competition/NeurIPS2023/sample/true_graph.npy')
# 导出 DataFrame 到 CSV 文件
true_graph_df = pd.DataFrame(true_graph)
print(true_graph_df)
csv_filename = 'W_true.csv'
true_graph_df.to_csv(csv_filename, index=False)

print(f"CSV 文件 {csv_filename} 已成功导出。")
