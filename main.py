from CD_methods import CD
import numpy as np
import debugpy
import os

datasets = [f'./datasets/dataset_{i}/alarm.csv' for i in range(1, 5)]
pri = [f'./datasets/dataset_{i}/causal_prior.npy' for i in range(1, 5)]
rca = [f'./datasets/dataset_{i}/rca_prior.csv' for i in range(1, 4)]
topo = [f'./datasets/dataset_{i}/topology.npy' for i in range(1, 4)]
rca.append(None)
topo.append(None)


def do_SAM():

    SAVE_PATH = [f'./SAM_submission/bin_form/dataset_{i}_graph_matrix.npy' for i in range(1, 5)]
    SAVE_PATH_w = [f'./SAM_submission/weighted_form/dataset_{i}_graph_matrix.npy' for i in range(1, 5)]
    k = -1

    for path in SAVE_PATH:
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            print("NotExist,Try to build a newone...")
            os.makedirs(directory)
            print("Built successfully")
    for path in SAVE_PATH_w:
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            print("NotExist,Try to build a newone...")
            os.makedirs(directory)
            print("Built successfully")

    for data_path in datasets:
        k += 1  

        cd = CD(origin_data_path=datasets[k], prior_imformation_path=pri[k], rca_prior_path=rca[k], topology_path=topo[k])

        est_matrix, weighted_est_matrix  = cd.SAM()
        np.save(SAVE_PATH[k], est_matrix)
        import os
        if not os.path.exists(SAVE_PATH_w[k]):            
            np.save(SAVE_PATH_w[k], weighted_est_matrix)
        print(f"Saved sucessfully in {SAVE_PATH[k]}!")

def do_NotearsNonlinear():  # 一直跑不出来

    SAVE_PATH = [f'./NotearsNonlinear_submission/dataset_{i}_graph_matrix.npy' for i in range(1, 5)]

    for path in SAVE_PATH:
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            print("NotExist,Try to build a newone...")
            os.makedirs(directory)
            print("Built successfully")

    k = -1
    for data_path in datasets:
        k += 1  
        cd = CD(origin_data_path=datasets[k], prior_imformation_path=pri[k], rca_prior_path=rca[k], topology_path=topo[k])
        est_matrix = cd.NotearsNonlinear()
        np.save(SAVE_PATH[k], est_matrix)
        print(f"Saved sucessfully in {SAVE_PATH[k]}!")
        print(f"----------------------------------------------dataset{k+1}--OK---------------NotearsNonlinear-----------------")

def do_TTPM():

    SAVE_PATH = [f'./TTPM100_submission/dataset_{i}_graph_matrix.npy' for i in range(1, 5)]

    for path in SAVE_PATH:
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            print("NotExist,Try to build a newone...")
            os.makedirs(directory)
            print("Built successfully")

    k = -1
    for data_path in datasets:
        k += 1  
        cd = CD(origin_data_path=datasets[k], prior_imformation_path=pri[k], rca_prior_path=rca[k], topology_path=topo[k])
        est_matrix = cd.TTPM()
        np.save(SAVE_PATH[k], est_matrix)
        print(f"Saved sucessfully in {SAVE_PATH[k]}!")
        print(f"----------------------------------------------dataset{k+1}--OK---------------TTPM----------------------------")

def do_PTHP():
    
    SAVE_PATH = [f'./PTHP80_submission/dataset_{i}_graph_matrix.npy' for i in range(1, 5)]

    for path in SAVE_PATH:
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            print("NotExist,Try to build a newone...")
            os.makedirs(directory)
            print("Built successfully")

    k = -1
    for data_path in datasets:
        k += 1  
        cd = CD(origin_data_path=datasets[k], prior_imformation_path=pri[k], rca_prior_path=rca[k], topology_path=topo[k])
        est_matrix = cd.PTHP()
        np.save(SAVE_PATH[k], est_matrix)
        print(f"Saved sucessfully in {SAVE_PATH[k]}!")
        print(f"----------------------------------------------dataset{k+1}--OK---------------PTHP----------------------------")


if __name__ == "__main__":
    # debugpy.connect(('192.168.1.50', 6789)) # 与跳板机链接，"192.168.1.50"是hpc跳板机内网IP，6789是跳板机接收调试信息的端口
    # debugpy.wait_for_client() # 等待跳板机的相应
    # debugpy.breakpoint() # 断点。一般而言直接在vscode界面上打断点即可。这个通过代码的方式提供一个断点，否则如果忘了在界面上打断点程序就会一直运行，除非点击停止按钮。

    #do_SAM()
    #do_NotearsNonlinear()
    do_PTHP()