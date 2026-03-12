from typing import List, Any
import networkx as nx
import copy
from pathlib import Path
import pandas as pd
from queue import Queue
from itertools import groupby
from operator import itemgetter
from time import time
from typing import Any


def get_list_G(dataset, ts, te):
    graph_path = './Data/' + dataset + '/'
    graph_files = list(Path(graph_path).glob('*'))[ts:te]
    list_G: list[Any] = [nx.read_gml(g) for g in graph_files]
    return list_G


def is_kcore(G, k):  # 判断是不是k-core
    if len(G.nodes) > 0:
        sorted_deg = sorted(G.degree, key=lambda x: x[1])
        return sorted_deg[0][1] >= k
    else:
        return False


def k_max(G):  # 返回最大core值节点
    return sorted(list(nx.core_number(G).items()), key=lambda x: x[1], reverse=True)[0]


def get_V_max(list_G, k):  # 所有图中k-core的最大节点值
    V_max = max([len(nx.k_core(g, k)) for g in list_G])
    return V_max


def remove_theta(G, theta):
    # return the graph that has filtered edges whose weights are smaller than theta
    G_temp = G.copy()
    for (u, v) in G_temp.edges:
        if G_temp[u][v]['weight'] < theta:
            G_temp.remove_edge(u, v)
    return G_temp


def local_k_core(G, k):
    max_k_core = nx.k_core(G, k)
    filtered_components = []
    for g in list(nx.connected_components(max_k_core)):
        filtered_components.append(nx.subgraph(max_k_core, g))
    return filtered_components


def G_induced_by_E_theta(G, theta):  # 边诱导图
    filtered_edges = [(u, v) for (u, v) in G.edges if G[u][v]['weight'] >= theta]
    H = G.edge_subgraph(filtered_edges)
    return H


def cal_S_rel(V_c, T_c, V_max, T_q, alpha):  # 得分函数#已
    aa = (1 + alpha * alpha) * (V_c / V_max * T_c / T_q) / (alpha * alpha * V_c / V_max + T_c / T_q)
    return aa


def filter_theta(G, theta):
    G_temp = G.copy()
    for (u, v) in G_temp.edges:
        # if G_temp[u][v]['weight'] * 10 < (theta + 1):
        if G_temp[u][v]['weight'] < theta:  # 直接比较，支持任意小数
            G_temp.remove_edge(u, v)
    return G_temp


def update_core_by_remove_theta(G, theta, df):
    origin_core = nx.core_number(G)
    G_temp = filter_theta(G, theta)
    filtered_core = nx.core_number(G_temp)
    for v, c in origin_core.items():
        if filtered_core[v] < c:
            for i in range(c - filtered_core[v]):
                # df.loc[(v, c - i)] = round(theta * 0.1, 2)
                df.loc[(v, c - i)] = theta
    return G_temp



def theta_table(G):
    k = k_max(G)[1]
    df = pd.DataFrame(index=sorted(G.nodes()),
                      columns=[f"k_{i}" for i in range(1, k + 1)])

    for theta in [round(x * 0.01, 2) for x in range(1, 101)]:  # 0.01到1.00
        G_filtered = filter_theta(G, theta)
        core_numbers = nx.core_number(G_filtered)
        for v, c in core_numbers.items():
            for k_val in range(1, c + 1):
                if pd.isna(df.at[v, f"k_{k_val}"]) or theta > df.at[v, f"k_{k_val}"]:
                    df.at[v, f"k_{k_val}"] = theta
    df = df.round(2).fillna(0)
    return df


# Construct the WCF-Index
class Node:
    def __init__(self, ids, list_v, theta):
        self.ids = ids
        self.vertex = list_v
        self.theta = theta
        self.parent = None
        self.children = set()

    def contains_v(self, v):
        return v in self.vertex

    def add_vertices(self, v):
        self.vertex.extend(v)

    def replace_vertices(self, v):
        self.vertex = v

    def remove_vertices(self, v):
        self.vertex = self.vertex.remove(v)

    def set_parent(self, p_Node_id):
        self.parent = p_Node_id

    def add_children(self, c_Node_id):
        self.children.add(c_Node_id)

    def remove_children(self, c_Node_id):
        self.children.remove(c_Node_id)

    def remove_parent(self):
        self.parent = None

    def get_root_in_tree(self, tree):
        if self.parent is None:
            return self
        p_id = self.parent
        p_node = tree[p_id]
        while p_node.parent:
            p_id = p_node.parent
            p_node = tree[p_id]
        return p_node

    def get_subgraph_in_tree(self, tree):
        visited = []
        all_nodes = []
        Q = Queue()
        Q.put(self.ids)
        while not Q.empty():
            X = Q.get()
            visited.append(X)
            all_nodes.extend(tree[X].vertex)
            for Y in tree[X].children:
                Q.put(Y)
        return all_nodes, visited

    def info(self):
        print('Node: {}\nverteices: {}\ntheta: {}\nparent: {}\nchildren: {}'.format(self.ids, self.vertex, self.theta,
                                                                                    self.parent, self.children))



def theta_tree(theta_thres_df, G):
    WCF_index = {}
    k_max_value = k_max(G)[1]  # 获取图中最大k-core值

    # 为每个k值构建theta树
    for k_curr in range(1, k_max_value + 1):
        theta_tree_k = {'theta': {}, 'node_id': {}}
        ids = 0
        merged_ids = set()

        # 获取当前k值对应的theta阈值列名
        col_name = f"k_{k_curr}"

        # 按theta值分组（降序）
        theta_groups = theta_thres_df.groupby(col_name)
        theta_v = sorted(theta_groups.groups.keys(), reverse=True)

        for theta in theta_v:
            theta = round(theta, 2)  # 确保精度
            theta_tree_k['theta'][theta] = []

            # 获取当前theta对应的节点列表
            node_v = theta_groups.get_group(theta).index.tolist()

            # 创建子图并过滤边
            sub_G = nx.subgraph(G, node_v)
            temp_G = remove_theta(sub_G, theta)

            # 处理连通分量
            for C in nx.connected_components(temp_G):
                merged = False
                X = Node(ids, list(C), theta)
                theta_tree_k['node_id'][ids] = X
                theta_tree_k['theta'][theta].append(ids)

                # 获取子图的邻居节点
                out_N = get_N_of_subgraph(nx.subgraph(G, C), G)
                visited = []

                # 检查邻居节点以确定合并
                for v in out_N:
                    if theta_thres_df.loc[v, col_name] > theta and not merged:
                        # 找到包含v的节点
                        nei = [y for y in theta_tree_k['node_id'].values() if y.contains_v(v)]
                        if nei and (nei[0] not in visited):
                            Y = nei[0]
                            visited.append(Y)
                            Z = Y.get_root_in_tree(theta_tree_k['node_id'])

                            if Z != X:
                                if Z.theta > X.theta:
                                    Z.set_parent(X.ids)
                                    X.add_children(Z.ids)
                                else:
                                    Z.add_vertices(X.vertex)
                                    for c_id in X.children:
                                        theta_tree_k['node_id'][c_id].set_parent(Z.ids)
                                        Z.add_children(c_id)
                                    if ids not in merged_ids:
                                        theta_tree_k['node_id'].pop(ids, None)
                                        theta_tree_k['theta'][theta].remove(ids)
                                        merged_ids.add(ids)
                                    merged = True
                ids += 1

        WCF_index[k_curr] = theta_tree_k

    return WCF_index


def get_N_of_subgraph(sub_G, G):
    # return neighbors of the subgraph
    node_N = set()
    for node in sub_G.nodes:
        node_N.update([i for i in G.neighbors(node)])
    return [i for i in node_N if i not in sub_G.nodes]


def is_root(tree, node, theta):
    res = False
    if tree['node_id'][node].theta >= theta:
        if tree['node_id'][node].parent is None:
            res = True

        elif tree['node_id'][tree['node_id'][node].parent].theta < theta:
            res = True
    return res


def return_C1(G, wcf_index, theta, k):
    theta = round(theta, 2)  # 标准化精度
    C_1 = []
    if k not in wcf_index.keys():
        return C_1

    tree = wcf_index[k]
    vertices = []
    S = [
        node
        for node in tree['node_id'].keys()
        if is_root(tree, node, theta)
    ]
    S = sorted(S, key=lambda x: tree['node_id'][x].theta)
    for root_id in S:
        C_vertices = tree['node_id'][root_id].get_subgraph_in_tree(tree['node_id'])[0]
        G_k_max = nx.subgraph(G, C_vertices)
        G_filtered = remove_theta(G_k_max, theta)
        components = local_k_core(G_filtered, k)
        C_1.extend(components)

    return C_1


def LCT(mu, M):
    maxLen = 0
    currLen = 0
    for value in M:
        if value >= mu:
            currLen += 1
        else:
            if currLen > maxLen:
                maxLen = currLen
            currLen = 0
    if currLen > maxLen:
        maxLen = currLen
    return maxLen


def UBR_wcf(T_i, L_c, V_max, T_q, alpha):
    UBR = float('-inf')
    for t in T_i:
        for component in L_c[1][t]:
            mu = len(component.nodes)
            score = cal_S_rel(mu, LCT(mu, [len(c.nodes) for c in L_c[1][t]]), V_max, T_q, alpha)
            UBR = max(UBR, score)
            return UBR


def find_component_intersection(components1, components2, k):
    intersection_list = []
    seen_graphs = set()
    for graph1 in components1:
        for graph2 in components2:
            intersection_graph = nx.intersection(graph1, graph2)
            if intersection_graph.number_of_nodes() > 0:
                intersection_graph_k = local_k_core(intersection_graph, k)
                for subgraph in intersection_graph_k:
                    # 使用图的节点集合和边集合的元组作为唯一标识符
                    graph_hash = (frozenset(subgraph.nodes), frozenset(subgraph.edges))
                    if graph_hash not in seen_graphs:
                        seen_graphs.add(graph_hash)
                        intersection_list.extend(intersection_graph_k)
    return intersection_list


def CRCP(list_G, WCF_indice, theta, k, V_max, alpha, r):
    T_q = len(list_G)
    all_size = [[0 for _ in range(len(list_G) + 1)] for _ in range(len(list_G) + 1)]
    L_c = [[[] for _ in range(len(list_G) + 1)] for _ in range(len(list_G) + 1)]
    score = [[[-1 for _ in range(0)] for _ in range(len(list_G) + 1)] for _ in range(len(list_G) + 1)]
    OptD = (-1, -1)
    anchored = []

    # 第一步：收集所有时间戳的初始组件
    for t, wcf_index in enumerate(WCF_indice, start=1):
        C_1 = return_C1(list_G[t - 1], wcf_index, theta, k)
        if C_1 is None:
            anchored.append(t)
        else:
            L_c[1][t].extend(C_1)

    # 第二步：处理时间序列上的组件延续性
    T_elig = [t + 1 for t in range(len(list_G)) if t + 1 not in anchored]
    T_s = []
    for _, g in groupby(enumerate(T_elig), lambda x: x[0] - x[1]):
        T_s.append(list(map(itemgetter(1), g)))

    # 第三步：计算所有候选组件的得分
    seen_components = {}

    def normalize_edges(edges):
        return {tuple(sorted(edge)) for edge in edges}

    all_ubr = {}
    for T_i in T_s:
        all_ubr[tuple(T_i)] = (UBR_wcf(T_i, L_c, V_max, T_q, alpha))

    sorted_T_s = sorted(all_ubr.keys(), key=lambda x: all_ubr[x], reverse=True)

    for T_i in sorted_T_s:
        for d in range(1, len(T_i) + 1):
            for t in T_i:
                if d <= (t - T_i[0] + 1):
                    if d > 1:
                        k_core_inter = find_component_intersection(L_c[d - 1][t - 1], L_c[d - 1][t], k)
                        if k_core_inter:
                            L_c[d][t] = k_core_inter

                    for component in L_c[d][t]:
                        # 标准化边表示
                        normalized_edges = normalize_edges(component.edges())
                        component_hash = (frozenset(component.nodes()), frozenset(normalized_edges))
                        component_score = cal_S_rel(len(component.nodes), d, V_max, T_q, alpha)

                        # 只保留得分更高的相同组件
                        if component_score > 0:
                            if component_hash not in seen_components or component_score > \
                                    seen_components[component_hash][0]:
                                seen_components[component_hash] = (component_score, component, d)

    # 第四步：排序并选择top-r
    unique_components = sorted(seen_components.values(), key=lambda x: x[0], reverse=True)
    # 确保不超过r个结果
    sorted_scores = [item[0] for item in unique_components[:r]]
    sorted_components = [item[1] for item in unique_components[:r]]

    return sorted_scores, sorted_components

