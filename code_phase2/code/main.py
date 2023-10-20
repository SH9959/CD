from CD_methods import CD
import numpy as np
import debugpy
import os
import logging
from datetime import datetime
import Utils
datasets = [f'./datasets/dataset_{i}/alarm.csv' for i in range(1, 5)]
pri = [f'./datasets/dataset_{i}/causal_prior.npy' for i in range(1, 5)]
rca = [f'./datasets/dataset_{i}/rca_prior.csv' for i in range(1, 4)]
topo = [f'./datasets/dataset_{i}/topology.npy' for i in range(1, 4)]
rca.append(None)
topo.append(None)
init_paths = [f'./init_matrix/init_d{i}.npy' for i in range(1, 5)]
SAVEPATH = "./submission_results"
# def do_SAM():
#
#     SAVE_PATH = [f'./SAM_submission/bin_form/dataset_{i}_graph_matrix.npy' for i in range(1, 5)]
#     SAVE_PATH_w = [f'./SAM_submission/weighted_form/dataset_{i}_graph_matrix.npy' for i in range(1, 5)]
#     k = -1
#
#     for path in SAVE_PATH:
#         Utils.check_path(path)
#
#     for path in SAVE_PATH_w:
#         Utils.check_path()
#
#     for data_path in datasets:
#         k += 1
#
#         cd = CD(origin_data_path=datasets[k], prior_imformation_path=pri[k], rca_prior_path=rca[k], topology_path=topo[k])
#
#         est_matrix, weighted_est_matrix  = cd.SAM()
#         np.save(SAVE_PATH[k], est_matrix)
#         import os
#         if not os.path.exists(SAVE_PATH_w[k]):
#             np.save(SAVE_PATH_w[k], weighted_est_matrix)
#         print(f"Saved sucessfully in {SAVE_PATH[k]}!")
#
# def do_NotearsNonlinear():  # 一直跑不出来
#
#     SAVE_PATH = [f'./NotearsNonlinear_submission/dataset_{i}_graph_matrix.npy' for i in range(1, 5)]
#
#     for path in SAVE_PATH:
#         Utils.check_path(path)
#
#     k = -1
#     for data_path in datasets:
#         k += 1
#         cd = CD(origin_data_path=datasets[k], prior_imformation_path=pri[k], rca_prior_path=rca[k], topology_path=topo[k])
#         est_matrix = cd.NotearsNonlinear()
#         np.save(SAVE_PATH[k], est_matrix)
#         print(f"Saved sucessfully in {SAVE_PATH[k]}!")
#         print(f"----------------------------------------------dataset{k+1}--OK---------------NotearsNonlinear-----------------")
#
# def do_TTPM():
#
#     SAVE_PATH = [f'./TTPM100_submission/dataset_{i}_graph_matrix.npy' for i in range(1, 5)]
#
#     for path in SAVE_PATH:
#         Utils.check_path(path)
#
#     k = -1
#     for data_path in datasets:
#         k += 1
#         cd = CD(origin_data_path=datasets[k], prior_imformation_path=pri[k], rca_prior_path=rca[k], topology_path=topo[k])
#         est_matrix = cd.TTPM()
#         np.save(SAVE_PATH[k], est_matrix)
#         print(f"Saved sucessfully in {SAVE_PATH[k]}!")
#         print(f"----------------------------------------------dataset{k+1}--OK---------------TTPM----------------------------")

def do_PTHP():
    
    SAVE_PATH = [f'./PTHP15_submission/dataset_{i}_graph_matrix.npy' for i in range(1, 5)]

    for path in SAVE_PATH:
        Utils.check_path(path)

    k = -1
    for data_path in datasets:
        k += 1  
        cd = CD(origin_data_path=datasets[k], prior_imformation_path=pri[k], rca_prior_path=rca[k], topology_path=topo[k])
        est_matrix = cd.PTHP()
        np.save(SAVE_PATH[k], est_matrix)
        print(f"Saved sucessfully in {SAVE_PATH[k]}!")
        print(f"----------------------------------------------dataset{k+1}--OK---------------PTHP----------------------------")
# def do_PTHP_MultiProcesses():  # 多进程
#     from multiprocessing import Pool
#     # 配置全局日志
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(message)s')
#
#     # pthp的参数
#     para = {
#         'delta':0.01,
#         'max_hop':2,
#         'penalty':'AIC',
#         'max_iter':100,
#         'epsilon':1
#     }
#
#
#         # 创建进程池
#
#     """
#     datasets = [f'./datasets/dataset_{i}/alarm.csv' for i in range(1, 5)]
#     pri = [f'./datasets/dataset_{i}/causal_prior.npy' for i in range(1, 5)]
#     rca = [f'./datasets/dataset_{i}/rca_prior.csv' for i in range(1, 4)]
#     topo = [f'./datasets/dataset_{i}/topology.npy' for i in range(1, 4)]
#     rca.append(None)
#     topo.append(None)
#     """
#     result_paths = [f'./PTHPs_results/PTHP_{para["max_iter"]}_Final_results/dataset_{i}_graph_matrix.npy' for i in range(1, 5)]
#     for path in result_paths:
#         Utils.check_path(path)
#
#     max_processes = 4  # 根据需要调整进程数
#
#     with Pool(max_processes) as pool:
#         pool.starmap(do_PTHP_single, zip(datasets, pri, rca, topo, [para] * 4, result_paths))
#
#     print("All datasets processed.")


def do_PTHP_single(d:str, p:str, r:str, t:str, para:dict, ini:str, savedir:str):  # 处理单个数据集,传入路径

# 输出进程信息
    n = Utils.get_task_id(d)
    
    logging.info(f"dataset_{n} Processing {d} in process {os.getpid()}")

            # 创建一个独立的日志文件，每个进程使用不同的文件名
                # 获取当前日期和时间
    current_datetime = datetime.now()

    # 将日期和时间格式化为字符串，精确到分钟
    now = current_datetime.strftime("%Y_%m_%d_%H_%M")

    log_file = f'dataset_{n}_pid{os.getpid()}_at_{now}.log'
    log_formatter = logging.Formatter('%(asctime)s - %(processName)s - %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(log_formatter)
    # 获取当前进程的日志记录器并添加处理程序
    local_logger = logging.getLogger()
    local_logger.addHandler(handler)

    # 输出进程信息
    local_logger.info(f"dataset_{n}-Processing_{d} in process {os.getpid()}")
    local_logger.info(f"initial parameter of PTHP{para}")
    logging.info(f"SAVE_DIRECTORY is in {savedir}")
    logging.info(f"prior_file_path:{p}")
    logging.info(f"rca_file_path:{r}")
    logging.info(f"topology_file_path:{t}")
    logging.info(f"init_prior_file_path:{ini}")
    logging.info(f"now:{now}")
    cd = CD(origin_data_path=d, prior_imformation_path=p, rca_prior_path=r, topology_path=t, pthp_para=para, init_re_path=ini, savedirpath=savedir)
    est_matrix = cd.PTHP()
    #np.save(s, est_matrix)
    return est_matrix


def do_one_PTHP(task:int):
   # task = 2
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(message)s')

    # pthp的参数
    para = {
        'delta':0.01, 
        'max_hop':2, 
        'penalty':'AIC', 
        'max_iter':75,
        'epsilon':1
    }

    save_dir = SAVEPATH
    Utils.check_path(save_dir)

    """
    datasets = [f'./datasets/dataset_{i}/alarm.csv' for i in range(1, 5)]
    pri = [f'./datasets/dataset_{i}/causal_prior.npy' for i in range(1, 5)]
    rca = [f'./datasets/dataset_{i}/rca_prior.csv' for i in range(1, 4)]
    topo = [f'./datasets/dataset_{i}/topology.npy' for i in range(1, 4)]
    rca.append(None)
    topo.append(None)
    """

    #result_paths = [f'./PTHPs_results_with_ini_918/PTHP_{para["max_iter"]}_Final_results/dataset_{i}_graph_matrix.npy' for i in range(1, 5)]

    do_PTHP_single(d=datasets[task], p=pri[task], r=rca[task], t=topo[task], para=para, ini=init_paths[task], savedir=save_dir)  # , ini = init_paths[task])



if __name__ == "__main__":
    # debugpy.connect(('192.168.1.50', 6789)) # 与跳板机链接，"192.168.1.50"是hpc跳板机内网IP，6789是跳板机接收调试信息的端口
    # debugpy.wait_for_client() # 等待跳板机的相应
    # debugpy.breakpoint() # 断点。一般而言直接在vscode界面上打断点即可。这个通过代码的方式提供一个断点，否则如果忘了在界面上打断点程序就会一直运行，除非点击停止按钮。

    #do_SAM()
    #do_NotearsNonlinear()
    #do_PTHP()
    # do_PTHP_MultiProcesses()
    TASKs = [1]
    for TASK in TASKs:
        do_one_PTHP(TASK - 1)
    # 可以并行执行四个数据集