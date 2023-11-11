import numpy as np
import pandas as pd
from copy import deepcopy
import time



class CD():

    def __init__(self, origin_data_path=None, prior_imformation_path=None, topology_path=None, rca_prior_path=None,
                 origin_data=None, prior_imformation=None, topology=None, rca_prior=None, init_re_path=None, pthp_para={'delta':0.01, 'max_hop':2, 'penalty':'BIC', 'max_iter':5, 'epsilon':1},
                 savedirpath = "./_PTHP_results") -> None:
        self.prior_imformation_path = prior_imformation_path
        self.origin_data_path = origin_data_path
        self.topology_path = topology_path
        self.rca_prior_path = rca_prior_path      
        self.pthp_para = pthp_para
        self.init_re_path = init_re_path
        self.savedirpath = savedirpath
        import re
        string = origin_data_path
        match = re.search(r'dataset_(\d+)', string)
        # 检查是否匹配到字符串
        if match:
            # 提取匹配到的数字部分并转换为整数
            number = int(match.group(1))
            print(number)  # 输出：1
        else:
            print("No match found in the string.")

        self.num = number  # 当前数据集名字（或者说编号）


        if self.origin_data_path is not None:
            self.origin_data = pd.read_csv(self.origin_data_path)
        else:
            self.origin_data = None

        if self.prior_imformation_path is not None:
            self.prior_imformation = np.load(self.prior_imformation_path)
        else:
            self.prior_imformation = np.zeros(shape=(max(self.origin_data["alarm_id"].values)+1, max(self.origin_data["alarm_id"].values)+1))
            self.prior_imformation.fill(-1)

        if self.topology_path is not None:
            self.topology = np.load(self.topology_path)

        else:
            self.topology = np.zeros(shape=(max(self.origin_data["device_id"].values)+1, max(self.origin_data["device_id"].values)+1))

        if self.rca_prior_path is not None:
            self.rca_prior = pd.read_csv(self.rca_prior_path)
        else:
            self.rca_prior = None
            self.pthp_para['max_hop'] = 1

        if self.init_re_path is not None:
            self.init_matrix = np.load(self.init_re_path)
        else:
            self.init_matrix = np.ones_like(self.prior_imformation)

    def origin_data_to_X(self, ):
        """
        官方给的原始数据转换得到X.csv
        """
        alarms = deepcopy(self.origin_data)
        TIME_WIN_SIZE  = 120
        #Use a time sliding window(300 seconds) to generate IID samples.
        alarms = alarms.sort_values(by='start_timestamp')
        alarms['win_id'] = alarms['start_timestamp'].map(lambda elem:int(elem/TIME_WIN_SIZE))

        samples=alarms.groupby(['alarm_id','win_id'])['start_timestamp'].count().unstack('alarm_id')
        samples = samples.dropna(how='all').fillna(0)
        samples = samples.sort_index(axis=1)
        print("samples.shape:", samples.shape)
        #np.save("X.csv", samples)  #暂时不保存了
        return samples


    def PC(self, ):
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
        causal_prior= deepcopy(self.prior_imformation)
        alarms = deepcopy(self.origin_data)
        print(f"shape of alarm data: {alarms.shape}")
        print(f"shape of causal prior matrix: {causal_prior.shape}")
        # Notes: topology.npy and rca_prior.csv are not used in this script.

        samples = self.origin_data_to_X()

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


        def SAM(self, threshold=0.35):  # 2022

            import networkx as nx  # 导入必要的函数包
            import scipy as sp
            import operator
            import matplotlib.pyplot as plp
            import cdt
            import networkx as nx
            import pandas as pd
            import matplotlib.pyplot as plt
            from cdt.causality.graph import SAM  # , PC,GES,LiNGAM,LiNGAM
            import numpy as np

            print("\nSAM IS RUNNING...")
            # Load the data
            # data = pd.read_csv('X.csv')
            data = self.origin_data_to_X()
            print(data, data.shape)
            # Infer the causal diagram
            A = time.time()

            _output = SAM().create_graph_from_data(data)
            # Visualize the diagram
            # pos = nx.circular_layout(_output)
            print(nx.adjacency_matrix(_output).todense())  # 返回图的邻接矩阵
            a = nx.adjacency_matrix(_output).todense()
            # np.save("weighted_SAM_est_graph.npy", a)

            B = time.time()
            print(f"time: {B-A}s")

            # threshold = 0.35  # 设置一个阈值
            binary_W_est = (a > threshold).astype(int)
            print("Binary W_est:\n", binary_W_est)

            # np.save("SAM_est_graph.npy", binary_W_est)
            print("Saved successfully as SAM_est_graph.npy!")
            # nx.draw_networkx(pc_output)# , pos)  # draw
            # plp.show()
            return (binary_W_est, a)

    def Notears(self, ):
        from trustworthyAI.gcastle.castle.algorithms import Notears
        from trustworthyAI.gcastle.castle.common.priori_knowledge import PrioriKnowledge
        """
        PC方法需要git clone trustworthyAI库,记得改绝对路径,
        dataset1: (2017,39) 8s
        dataset2: (2016,49) 37s
        dataset3: (2017,31) 3s
        dataset4: (6582,30) 3000s
        """
        print(" ")
        print("Notears is running")
        t1 = time.time()

        # causal_prior
        causal_prior = deepcopy(self.prior_imformation)
        alarms = deepcopy(self.origin_data)
        print(f"shape of alarm data: {alarms.shape}")
        print(f"shape of causal prior matrix: {causal_prior.shape}")
        # Notes: topology.npy and rca_prior.csv are not used in this script.

        samples = self.origin_data_to_X()

        # create the prior knowledge object for the PC algorithm
        prior_knowledge = PrioriKnowledge(causal_prior.shape[0])
        for i, j in zip(*np.where(causal_prior == 1)):
            prior_knowledge.add_required_edge(i, j)

        for i, j in zip(*np.where(causal_prior == 0)):
            prior_knowledge.add_forbidden_edge(i, j)

        pc = Notears()
        pc.learn(samples)
        graph_matrix = np.array(pc.causal_matrix)
        print("graph_matrix.shape:", graph_matrix.shape)
        t2 = time.time()

        print(f"time: {t2 - t1}s")
        return graph_matrix

    def NotearsNonlinear(self,): # 2018-
        from trustworthyAI.gcastle.castle.algorithms import NotearsNonlinear
        from trustworthyAI.gcastle.castle.common.priori_knowledge import PrioriKnowledge
        """
        PC方法需要git clone trustworthyAI库,记得改绝对路径,
        dataset1: (2017,39) 
        dataset2: (2016,49) 
        dataset3: (2017,31) 
        dataset4: (6582,30)
        """
        print(" ")
        print("NotearsNonlinear is running")
        t1 = time.time()

        # causal_prior
        causal_prior= deepcopy(self.prior_imformation)
        alarms = deepcopy(self.origin_data)
        print(f"shape of alarm data: {alarms.shape}")
        print(f"shape of causal prior matrix: {causal_prior.shape}")
        # Notes: topology.npy and rca_prior.csv are not used in this script.

        samples = self.origin_data_to_X()

        # create the prior knowledge object for the PC algorithm 



        nnl = NotearsNonlinear(device_type="gpu")
        nnl.learn(samples)
        graph_matrix = np.array(nnl.causal_matrix)


        #  add priorknowledge
        prior_knowledge = PrioriKnowledge(causal_prior.shape[0])
        for i, j in zip(*np.where(causal_prior == 1)):
            prior_knowledge.add_required_edge(i, j)
            graph_matrix[i][j] = 1

        for i, j in zip(*np.where(causal_prior == 0)):
            prior_knowledge.add_forbidden_edge(i, j)
            graph_matrix[i][j] = 0

        print("graph_matrix.shape:", graph_matrix.shape)
        print("Nontears_Nonlinear_est_map:", graph_matrix)
        t2 = time.time()

        print(f"time: {t2-t1}s")

        return graph_matrix


    def GES(self,):  # 2006 不跑了，需要R
        import networkx as nx  # 导入必要的函数包
        import scipy as sp
        import operator
        import matplotlib.pyplot as plp
        import cdt
        import networkx as nx
        import pandas as pd
        import matplotlib.pyplot as plt
        from cdt.causality.graph import GES  # PC,LiNGAM,LiNGAM,SAM
        import numpy as np
        print("\nGES IS RUNNING...")
        # Load the data
        # data = pd.read_csv('X.csv')
        data = self.origin_data_to_X()
        print(data, data.shape)
        # Infer the causal diagram
        A = time.time()

        _output = GES().create_graph_from_data(data)
        # Visualize the diagram
        # pos = nx.circular_layout(_output)
        print(nx.adjacency_matrix(_output).todense())  # 返回图的邻接矩阵
        a = nx.adjacency_matrix(_output).todense()
        np.save("weighted_GES_est_graph.npy", a)

        B = time.time()
        print(f"time: {B-A}s")
        threshold = 0.35  # 设置一个阈值
        binary_W_est = (a > threshold).astype(int)
        print("Binary W_est:\n", binary_W_est)

        # np.save("GES_est_graph.npy", binary_W_est)
        print("Saved successfully as GES_est_graph.npy!")
        # nx.draw_networkx(pc_output)# , pos)  # draw
        # plp.show()
        return (binary_W_est, a)
        
    def NTS_NOTEARS(elf, ):  # 一种加入了先验约束的模型

        pass

    def DYNOTEARS(self, ):

        pass

    def TTPM(self, ):  # 当前问题，太久

        from trustworthyAI.gcastle.castle.algorithms import TTPM
        from trustworthyAI.gcastle.castle.common.priori_knowledge import PrioriKnowledge
        """
        PC方法需要git clone trustworthyAI库,记得改绝对路径,
        dataset1: (2017,39) 8s
        dataset2: (2016,49) 37s
        dataset3: (2017,31) 3s
        dataset4: (6582,30) 3000s
        """
        print(" ")
        print("TTPM is running")
        t1 = time.time()

        # causal_prior
        causal_prior= deepcopy(self.prior_imformation)

        alarm_data = deepcopy(self.origin_data)

        print(f"shape of alarm data: {alarm_data.shape}")
        print(f"shape of causal prior matrix: {causal_prior.shape}")
        # Notes: topology.npy and rca_prior.csv are not used in this script.

        # samples = self.origin_data_to_X()

        #alarm_data = pd.read_csv(path+str(self.num)+'/Alarm.csv', encoding='utf')alarm_data - alarm_data.iloc[ : , 0:3]

        alarm_data = alarm_data.drop(alarm_data.columns[3], axis=1)
        alarm_data.columns = ['event', 'node', 'timestamp']

        alarm_data[['event', 'node', 'timestamp']] = alarm_data[['event', 'timestamp', 'node']]
        alarm_data.columns = ['event', 'timestamp', 'node']


        count = 0 # 数据的前count行
        if count == 0:  # 选择全部
            alarm_data = alarm_data.iloc[ :, :]
        else:
            alarm_data = alarm_data.iloc[ : count,: ]

        samples = alarm_data

        if (self.topology == None).all():
            self.topology = np.zeros(shape=(max(alarm_data["node"].values)+1, max(alarm_data["node"].values)+1))

        # create the prior knowledge object for the PC algorithm 
        prior_knowledge = PrioriKnowledge(causal_prior.shape[0])
        for i, j in zip(*np.where(causal_prior == 1)):
            prior_knowledge.add_required_edge(i, j)

        for i, j in zip(*np.where(causal_prior == 0)):
            prior_knowledge.add_forbidden_edge(i, j)

        ttpm = TTPM(topology_matrix=self.topology, delta=0.01, max_hop=2, penalty='BIC', max_iter=100, epsilon=1)
        #  参数设置参照往年冠军设置

        ttpm.learn(samples)
        graph_matrix = np.array(ttpm.causal_matrix)
        print("graph_matrix.shape:", graph_matrix.shape)
        t2 = time.time()

        print(f"time: {t2-t1}s")
        return graph_matrix

    def SHP(self, ):

        pass
    

    def PTHP(self, ):
        from pthp.code.castle_mod.algorithms import PTHP

        # from pthp.code.castle_mod.algorithms import PTHP
        from trustworthyAI.gcastle.castle.common.priori_knowledge import PrioriKnowledge
        """
        方法需要git clone trustworthyAI库,记得改绝对路径,
        dataset1: (2017,39) 8s
        dataset2: (2016,49) 37s
        dataset3: (2017,31) 3s
        dataset4: (6582,30) 3000s
        """
        print("\nPTHP is running")
        t1 = time.time()

        # causal_prior
        causal_prior = deepcopy(self.prior_imformation)

        alarm_data = deepcopy(self.origin_data)

        print(f"shape of alarm data: {alarm_data.shape}")
        print(f"shape of causal prior matrix: {causal_prior.shape}")
        # Notes: topology.npy and rca_prior.csv are not used in this script.

        # samples = self.origin_data_to_X()

        #alarm_data = pd.read_csv(path+str(self.num)+'/Alarm.csv', encoding='utf')alarm_data - alarm_data.iloc[ : , 0:3]

        alarm_data = alarm_data.drop(alarm_data.columns[3], axis=1)
        alarm_data.columns = ['event', 'node', 'timestamp']

        alarm_data[['event', 'node', 'timestamp']] = alarm_data[['event', 'timestamp', 'node']]
        alarm_data.columns = ['event', 'timestamp', 'node']


        count = 0 # 数据的前count行
        if count == 0:  # 选择全部
            alarm_data = alarm_data.iloc[ :, :]
        else:
            alarm_data = alarm_data.iloc[ : count,: ]

        samples = alarm_data

        # create the prior knowledge object for the PC algorithm 
        # prior_knowledge = PrioriKnowledge(causal_prior.shape[0])
        # for i, j in zip(*np.where(causal_prior == 1)):
        #     prior_knowledge.add_required_edge(i, j)

        # for i, j in zip(*np.where(causal_prior == 0)):
        #     prior_knowledge.add_forbidden_edge(i, j)

        pthp = PTHP(topology_matrix=self.topology, prior_matrix=causal_prior, init_result_matrix=self.init_matrix, delta=self.pthp_para['delta'], max_hop=self.pthp_para['max_hop'], penalty=self.pthp_para['penalty'], max_iter=self.pthp_para['max_iter'], epsilon=self.pthp_para['epsilon'], dataname=self.num, save_dir_path=self.savedirpath)
        #  参数设置参照往年冠军设置
        # print("debug here1")
        # print("learning...")
        pthp.learn(samples)
        # print("ok")
        graph_matrix = np.array(pthp.causal_matrix)

        np.fill_diagonal(graph_matrix , 0)  # 非循环

        print("graph_matrix.shape:", graph_matrix.shape)
        print("est_graph:", graph_matrix)
        t2 = time.time()

        print(f"time: {t2-t1}s")
        return graph_matrix
    

def check_file_existence(file_path):
    import os
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'FileNotExist：{file_path}')
        return False
    return True

if __name__ == "__main__":
    pass