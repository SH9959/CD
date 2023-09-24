# coding=utf-8
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import pandas as pd
import numpy as np
import networkx as nx
import copy
from itertools import product
import concurrent.futures
import os
from trustworthyAI.gcastle.castle.common import BaseLearner

import Utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class PTHP(BaseLearner):  # 添加了一个数据集的名字参数，用来保存数据集，具体更改在79,96,248行
    """
    PTHP Algorithm.

    A causal structure learning algorithm based on Topological Hawkes process
     for spatio-temporal event sequences.

    Parameters
    ----------
    topology_matrix: np.matrix
        Interpreted as an adjacency matrix to generate the graph.
        It should have two dimensions, and should be square.

    prior_matrix: np.matrix
        the matrix of prior information

    delta: float, default=0.1
            Time decaying coefficient for the exponential kernel.

    epsilon: int, default=1
        BIC penalty coefficient.

    max_hop: positive int, default=6
        The maximum considered hops in the topology,
        when ``max_hop=0``, it is divided by nodes, regardless of topology.

    penalty: str, default=BIC
        Two optional values: 'BIC' or 'AIC'.
        
    max_iter: int
        Maximum number of iterations.

    Examples
    --------
    >>> from castle.common import GraphDAG
    >>> from castle.metrics import MetricsDAG
    >>> from castle.datasets import load_dataset
    >>> from castle.algorithms import TTPM
    # Data Simulation for TTPM
    >>> true_causal_matrix, topology_matrix, X = load_dataset(name='thp_test')
    >>> ttpm = TTPM(topology_matrix, max_hop=2)
    >>> ttpm.learn(X)
    >>> causal_matrix = ttpm.causal_matrix
    # plot est_dag and true_dag
    >>> GraphDAG(ttpm.causal_matrix.values, true_causal_matrix)
    # calculate accuracy
    >>> ret_metrix = MetricsDAG(ttpm.causal_matrix.values, true_causal_matrix)
    >>> ret_metrix.metrics
    """

    def __init__(self, topology_matrix, prior_matrix, PC_result_matrix = None,delta=0.1, epsilon=1,
                 max_hop=0, penalty='BIC', max_iter=20, dataname=0, save_dir_path="./_PTHP_results"):
        BaseLearner.__init__(self)
        assert isinstance(topology_matrix, np.ndarray),\
            'topology_matrix should be np.matrix object'
        assert topology_matrix.ndim == 2,\
            'topology_matrix should be two dimension'
        assert topology_matrix.shape[0] == topology_matrix.shape[1],\
            'The topology_matrix should be square.'
        self._topo = nx.from_numpy_array(topology_matrix, create_using=nx.Graph)

        # initialize instance variables
        self._penalty = penalty
        self._delta = delta
        self._max_hop = max_hop
        self._epsilon = epsilon
        self._max_iter = max_iter
        self._prior_matrix = prior_matrix
        self._dataname = dataname
        self._PC_result_matrix = PC_result_matrix  # 添加PC_result
        if PC_result_matrix is None:
            self._PC_result_matrix = np.ones_like(self._prior_matrix)
        self._save_dir_path = save_dir_path
    def learn(self, tensor, *args, **kwargs):
        """
        Set up and run the TTPM algorithm.

        Parameters
        ----------
        tensor:  pandas.DataFrame
            (V 1.0.0, we'll eliminate this constraint in the next version)
            The tensor is supposed to contain three cols:
                ['event', 'timestamp', 'node']

            Description of the three columns:
                event: event name (type).
                timestamp: occurrence timestamp of event, i.e., '1615962101.0'.
                node: topological node where the event happened.
        """

        # data type judgment
        if not isinstance(tensor, pd.DataFrame):
            raise TypeError('The tensor type is not correct,'
                            'only receive pd.DataFrame type currently.')

        cols_list = ['event', 'timestamp', 'node']
        for col in cols_list:
            if col not in tensor.columns:
                raise ValueError(
                    "The data tensor should contain column with name {}".format(
                        col))

        # initialize needed values
        self._start_init(tensor)

        # Generate causal matrix (DAG)
        _, raw_causal_matrix = self._hill_climb()
        self._causal_matrix = pd.DataFrame(raw_causal_matrix,
                                           index=self._matrix_names,
                                           columns=self._matrix_names)

    def _start_init(self, tensor):
        """
        Generates some required initial values.
        """
        tensor.dropna(axis=0, how='any', inplace=True)
        tensor['timestamp'] = tensor['timestamp'].astype(float)

        tensor = tensor.groupby(
            ['event', 'timestamp', 'node']).apply(len).reset_index()
        tensor.columns = ['event', 'timestamp', 'node', 'times']
        tensor = tensor.reindex(columns=['node', 'timestamp', 'event', 'times'])

        tensor = tensor.sort_values(['node', 'timestamp'])
        self.tensor = tensor[tensor['node'].isin(self._topo.nodes)]

        # calculate considered events
        self._event_names = np.array(list(set(self.tensor['event'])))
        self._event_names.sort()
        self._N = len(self._event_names)
        self._matrix_names = list(self._event_names.astype(str))

        # map event name to corresponding index value
        self._event_indexes = self._map_event_to_index(
            self.tensor['event'].values, self._event_names)
        self.tensor['event'] = self._event_indexes

        self._g = self._topo.subgraph(self.tensor['node'].unique())
        self._ne_grouped = self.tensor.groupby('node')

        self._decay_effects = np.zeros(
            [len(self._event_names), self._max_hop+1])  # will be used in EM.

        self._max_s_t = tensor['timestamp'].max()
        self._min_s_t = tensor['timestamp'].min()

        for k in range(self._max_hop+1):
            self._decay_effects[:, k] = tensor.groupby('event').apply(
                lambda i: ((((1 - np.exp(
                    -self._delta * (self._max_s_t - i['timestamp']))) / self._delta)
                            * i['times']) * i['node'].apply(
                    lambda j: len(self._k_hop_neibors(j, k)))).sum())
        # |V|x|T|
        self._T = (self._max_s_t - self._min_s_t) * len(tensor['node'].unique())

    def _k_hop_neibors(self, node, k):

        if k == 0:
            return {node}
        else:
            return set(nx.single_source_dijkstra_path_length(
                self._g, node, k).keys()) - set(
                nx.single_source_dijkstra_path_length(
                    self._g, node, k - 1).keys())

    @staticmethod
    def _map_event_to_index(event_names, base_event_names):
        """
        Maps the event name to the corresponding index value.

        Parameters
        ----------
        event_names: np.ndarray, shape like (52622,)
            All occurred event names sorted by node and timestamp.
        base_event_names: np.ndarray, shape like (10,)
            All deduplicated and sorted event names

        Returns
        -------
        np.ndarray: All occurred event names mapped to their corresponding index 
         in base_event_names.
        """
        return np.array(list(map(lambda event_name:
                                 np.where(base_event_names == event_name)[0][0],
                                 event_names)))
    # 定义一个函数来执行单个 EM 计算任务
    def calculate_em_task(self, new_edge_mat):
        new_result = self._em(new_edge_mat)  # 在这里执行 EM 计算任务
        return [new_result, new_edge_mat]

    def _hill_climb_boost(self):  # 使用并发技术的加速代码
        """
        Search the best causal graph, then generate the causal matrix (DAG).

        Returns
        -------
        result: tuple, (likelihood, alpha matrix, events vector)
            likelihood: used as the score criteria for searching the
                causal structure.
            alpha matrix: the intensity of causal effect from event v’ to v.
            events vector: the exogenous base intensity of each event.
        edge_mat: np.ndarray
            Causal matrix.
        """
        self._get_effect_tensor_decays()
        # Initialize the adjacency matrix
        edge_mat = ((self._prior_matrix.copy() == 1) + np.eye(self._N, self._N)).astype(int)
        # input(edge_mat)
        # edge_mat = np.load("D:/Study/code/python_study/python_/PCIC/phase2/testing/1_notopo_5000_0.01_1.npy")
        # np.fill_diagonal(edge_mat, 1)
        # edge_mat[22][13] = 0
        # edge_mat[13][9] = 1
        # edge_mat[11][1] = 1
        # edge_mat[10][4] = 1
        # edge_mat[8][10] = 1
        # edge_mat[8][7] = 1
        # edge_mat[7][10] = 0
        # edge_mat[2][4] = 1
        # edge_mat[16][9] = 1
        result = self._em(edge_mat)
        l_ret = result[0]
        # 创建锁对象用于线程同步
        num_threads = 80  # 假设使用 4 个线程
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for num_iter in range(self._max_iter):
                
                logging.info('[iter {}]: likelihood_score = {}'.format(num_iter, l_ret))
                if num_iter % 2 == 0 and num_iter != 0:
                    # 每2代保存一下
                    # logging.info(f'edge_mat:{edge_mat}')
                    pa = os.path.join(self._save_dir_path, f"PTHP_{num_iter}_results", f"dataset_{self._dataname}_graph_matrix.npy")
                    #pa = f"{self.save_dir_path}/PTHP_{num_iter}_results/dataset_{self._dataname}_graph_matrix.npy"
                    Utils.check_path(pa)
                    tmp_mat = copy.deepcopy(edge_mat)
                    tmp_mat = Utils.filter(wait_to_filter_matrix=tmp_mat, prior_matrix=self._prior_matrix)
                    logging.info(f'tmp_edge_mat:{tmp_mat}')
                    np.save(pa, tmp_mat)
                    logging.info(f"iter[{num_iter}]:saved-----------in---------------{self._save_dir_path}------------------")
                # stop_tag = True
                neighbors = list(self._one_step_change_iterator(edge_mat))
                task_results = executor.map(self.calculate_em_task, neighbors)
                        # 等待所有任务完成
                concurrent.futures.wait(neighbors)
                # 等待所有任务完成并获取结果
                results = list(task_results)

                # 选择最优的结果
                best_result = max(results, key=lambda result: result[0][0])

                # 获取最优的似然分数和图
                new_result = best_result[0]
                new_edge_mat = best_result[0][1]

                if new_result[0] <= l_ret:
                #    stop_tag = True
                    break
                else:
                    edge_mat = new_edge_mat
                    result = new_result

                # if stop_tag:
                #     break

        return result, edge_mat
    
    def _hill_climb(self):
        """
        Search the best causal graph, then generate the causal matrix (DAG).

        Returns
        -------
        result: tuple, (likelihood, alpha matrix, events vector)
            likelihood: used as the score criteria for searching the
                causal structure.
            alpha matrix: the intensity of causal effect from event v’ to v.
            events vector: the exogenous base intensity of each event.
        edge_mat: np.ndarray
            Causal matrix.
        """
        self._get_effect_tensor_decays()
        # Initialize the adjacency matrix
        edge_mat = ((self._prior_matrix.copy() == 1) + np.eye(self._N, self._N)).astype(int)
        # input(edge_mat)
        # edge_mat = np.load("D:/Study/code/python_study/python_/PCIC/phase2/testing/1_notopo_5000_0.01_1.npy")
        # np.fill_diagonal(edge_mat, 1)
        # edge_mat[22][13] = 0
        # edge_mat[13][9] = 1
        # edge_mat[11][1] = 1
        # edge_mat[10][4] = 1
        # edge_mat[8][10] = 1
        # edge_mat[8][7] = 1
        # edge_mat[7][10] = 0
        # edge_mat[2][4] = 1
        # edge_mat[16][9] = 1
        result = self._em(edge_mat)
        l_ret = result[0]
        
        for num_iter in range(self._max_iter):
            
            logging.info('[iter {}]: likelihood_score = {}'.format(num_iter, l_ret))
            if num_iter % 2 == 0 and num_iter != 0:
                # 每2代保存一下

                pa = os.path.join(self._save_dir_path, f"PTHP_{num_iter}_results", f"dataset_{self._dataname}_graph_matrix.npy")
                #pa = f"./PTHPs_results_with_PC_918/PTHP_{num_iter}_results/dataset_{self._dataname}_graph_matrix.npy"
                Utils.check_path(pa)
                tmp_mat = copy.deepcopy(edge_mat)
                tmp_mat = Utils.filter(wait_to_filter_matrix=tmp_mat, prior_matrix=self._prior_matrix)
                logging.info(f'tmp_edge_mat:{tmp_mat}')
                np.save(pa, tmp_mat)
                #logging.info(f"iter[{num_iter}]:saved--------")
                logging.info(f"iter[{num_iter}]:saved-----------in---------------{self._save_dir_path}------------------")
            stop_tag = True
            for new_edge_mat in list(
                    self._one_step_change_iterator(edge_mat)):
                new_result = self._em(new_edge_mat)
                new_l = new_result[0]
                # Termination condition:
                #   no adjacency matrix with higher likelihood appears
                if new_l > l_ret:
                    result = new_result
                    l_ret = new_l
                    stop_tag = False
                    edge_mat = new_edge_mat

            if stop_tag:
                return result, edge_mat
        
        return result, edge_mat

    def _get_effect_tensor_decays(self):

        self._effect_tensor_decays = np.zeros([self._max_hop+1,
                                               len(self.tensor),
                                               len(self._event_names)])
        for k in range(self._max_hop+1):
            self._get_effect_tensor_decays_each_hop(k)

    def _get_effect_tensor_decays_each_hop(self, k):

        j = 0
        pre_effect = np.zeros(self._N)
        tensor_array = self.tensor.values
        for item_ind in range(len(self.tensor)):
            sub_n, start_t, ala_i, times = tensor_array[
                item_ind, [0, 1, 2, 3]]
            last_sub_n, last_start_t, last_ala_i, last_times = \
                tensor_array[item_ind - 1, [0, 1, 2, 3]]
            if (last_sub_n != sub_n) or (last_start_t > start_t):
                j = 0
                pre_effect = np.zeros(self._N)
                try:
                    k_hop_neighbors_ne = self._k_hop_neibors(sub_n, k)
                    neighbors_table = pd.concat(
                        [self._ne_grouped.get_group(i)
                         for i in k_hop_neighbors_ne])
                    neighbors_table = neighbors_table.sort_values(
                        'timestamp')
                    neighbors_table_value = neighbors_table.values
                except ValueError as e:
                    k_hop_neighbors_ne = []

                if len(k_hop_neighbors_ne) == 0:
                    continue

            cur_effect = pre_effect * np.exp(
                (np.min((last_start_t - start_t, 0))) * self._delta)
            while 1:
                try:
                    nei_sub_n, nei_start_t, nei_ala_i, nei_times \
                        = neighbors_table_value[j, :]
                except:
                    break
                if nei_start_t < start_t:
                    cur_effect[int(nei_ala_i)] += nei_times * np.exp(
                        (nei_start_t - start_t) * self._delta)
                    j += 1
                else:
                    break
            pre_effect = cur_effect

            self._effect_tensor_decays[k, item_ind] = pre_effect

    def _em(self, edge_mat):
        """
        E-M module, used to find the optimal parameters.

        Parameters
        ----------
        edge_mat： np.ndarray
            Adjacency matrix.

        Returns
        -------
        likelihood: used as the score criteria for searching the
            causal structure.
        alpha matrix: the intensity of causal effect from event v’ to v.
        events vector: the exogenous base intensity of each event.
        """

        causal_g = nx.from_numpy_array((edge_mat - np.eye(self._N, self._N)),
                                        create_using=nx.DiGraph)

        if not nx.is_directed_acyclic_graph(causal_g):
            return -100000000000000, \
                   np.zeros([len(self._event_names), len(self._event_names)]), \
                   np.zeros(len(self._event_names))

        # Initialize alpha:(nxn)，mu:(nx1) and L
        alpha = np.ones([self._max_hop+1, len(self._event_names),
                         len(self._event_names)])
        alpha = alpha * edge_mat
        mu = np.ones(len(self._event_names))
        l_init = 0

        for i in range(len(self._event_names)):
            pa_i = set(np.where(edge_mat[:, i] == 1)[0])
            li = -100000000000
            ind = np.where(self._event_indexes == i)
            x_i = self.tensor['times'].values[ind]
            x_i_all = np.zeros_like(self.tensor['times'].values)
            x_i_all[ind] = x_i
            while 1:
                # Calculate the first part of the likelihood
                lambda_i_sum = (self._decay_effects
                                * alpha[:, :, i].T).sum() + mu[i] * self._T

                # Calculate the second part of the likelihood
                lambda_for_i = np.zeros(len(self.tensor)) + mu[i]
                for k in range(self._max_hop+1):
                    lambda_for_i += np.matmul(
                        self._effect_tensor_decays[k, :],
                        alpha[k, :, i].T)
                lambda_for_i = lambda_for_i[ind]
                x_log_lambda = (x_i * np.log(lambda_for_i)).sum()
                new_li = -lambda_i_sum + x_log_lambda

                # Iteration termination condition
                delta = new_li - li
                if delta < 0.1:
                    li = new_li
                    l_init += li
                    pa_i_alpha = dict()
                    for j in pa_i:
                        pa_i_alpha[j] = alpha[:, j, i]
                    break
                li = new_li
                # update mu
                mu[i] = ((mu[i] / lambda_for_i) * x_i).sum() / self._T
                # update alpha
                for j in pa_i:
                    for k in range(self._max_hop+1):
                        upper = ((alpha[k, j, i] * (
                            self._effect_tensor_decays[k, :, j])[ind]
                                  / lambda_for_i) * x_i).sum()
                        lower = self._decay_effects[j, k]
                        if lower == 0:
                            alpha[k, j, i] = 0
                            continue
                        alpha[k, j, i] = upper / lower
            i += 1

        if self._penalty == 'AIC':
            return l_init - (len(self._event_names)
                             + self._epsilon * edge_mat.sum()
                             * (self._max_hop+1)), alpha, mu
        elif self._penalty == 'BIC':
            return l_init - (len(self._event_names)
                             + self._epsilon * edge_mat.sum()
                             * (self._max_hop+1)) * np.log(
                self.tensor['times'].sum()) / 2, alpha, mu
        else:
            raise ValueError("The penalty's value should be BIC or AIC.")

    def _one_step_change_iterator(self, edge_mat):

        return map(lambda e: self._one_step_change(edge_mat, e), self.check_prior())

    def check_prior(self):
        return [(i, j) for i, j in product(range(len(self._event_names)),
                           range(len(self._event_names))) if self._prior_matrix[i][j] == -1 and self._PC_result_matrix[i][j] == 1]


    @staticmethod
    def _one_step_change(edge_mat, e):
        """
        Changes the edge value in the edge_mat.

        Parameters
        ----------
        edge_mat: np.ndarray
            Adjacency matrix.
        e: tuple_like (j,i)

        Returns
        -------
        new_edge_mat: np.ndarray
            new value of edge
        """
        # logging.info(f"j,i:{e}--")
        j, i = e
        if j == i:
            return edge_mat
        new_edge_mat = edge_mat.copy()

        if new_edge_mat[j, i] == 1:
            new_edge_mat[j, i] = 0
            return new_edge_mat
        else:
            new_edge_mat[j, i] = 1
            new_edge_mat[i, j] = 0
            return new_edge_mat
