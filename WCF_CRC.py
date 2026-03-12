import networkx as nx
import copy
from pathlib import Path
from collections import deque
import pandas as pd
from queue import Queue
from itertools import groupby
from operator import itemgetter
from heapq import heappush, heappushpop
import time


# ================ Utilities ========================
def get_list_G(dataset, ts, te):
    graph_path = './Data/' + dataset + '/'
    graph_files = list(Path(graph_path).glob('*'))[ts:te]
    list_G = [nx.read_gml(g) for g in graph_files]
    return list_G


def is_kcore(G, k):
    if len(G.nodes) > 0:
        sorted_deg = sorted(G.degree, key=lambda x: x[1])
        return sorted_deg[0][1] >= k
    else:
        return False


def k_max(G):
    return sorted(list(nx.core_number(G).items()), key=lambda x: x[1], reverse=True)[0]


def remove_theta(G, theta, query=None):
    # return the graph that has filtered edges whose weights are smaller than theta
    G_temp = G.copy()
    for (u, v) in G_temp.edges:
        if G_temp[u][v]['weight'] < theta:
            G_temp.remove_edge(u, v)
    if query:
        for g in list(nx.connected_components(G_temp)):
            if query in g:
                G_temp = nx.subgraph(G_temp, g)
    return G_temp


def local_k_core(G, query, k):
    max_k_core = nx.k_core(G, k)
    filtered_G = nx.Graph()
    for g in list(nx.connected_components(max_k_core)):
        if query in g:
            filtered_G = nx.subgraph(max_k_core, g)
    return filtered_G


def get_G_max(list_G, query, theta, k, filtered=False):
    if filtered:
        G_max = [local_k_core(g, query, k) for g in list_G]
    else:
        G_max = [local_k_core(remove_theta(g, query, theta), query, k) for g in list_G]
    return G_max


def get_V_max(list_G, k):
    V_max = -1
    V_max = max([len(nx.k_core(g, k)) for g in list_G])
    return V_max


def G_induced_by_E_theta(G, theta):
    filtered_edges = [(u, v) for (u, v) in G.edges if G[u][v]['weight'] >= theta]
    H = G.edge_subgraph(filtered_edges)
    return H


def cal_S_rel(V_c, T_c, V_max, T_q, alpha):
    aa = (1 + alpha * alpha) * (V_c / V_max * T_c / T_q) / (alpha * alpha * V_c / V_max + T_c / T_q)
    return aa


# ================ EEF (Modified) ========================


def bfs_lambda_theta(list_G, theta, k, V_max, source=None):
    def edge_id(edge):
        return (frozenset(edge[:2]),) + edge[2:]

    lambda_theta = {}
    UBR = {}
    for t, G in enumerate(list_G, start=1):
        lambda_theta[t] = {}
        UBR.setdefault(t, {})
        visited_nodes = {source}
        visited_edges = set()
        queue = deque([(source, list(G.edges(source)))])
        while queue:
            parent, children_edges = queue.popleft()
            for edge in children_edges:
                child = edge[1]
                if (child not in visited_nodes) and (child in G.nodes):
                    if G.degree[child] >= k:
                        visited_nodes.add(child)
                        queue.append((child, list(G.edges(child))))
                    else:
                        G.remove_node(child)
                        continue
                edgeid = edge_id(edge)
                if edgeid not in visited_edges and edge in G.edges:
                    visited_edges.add(edgeid)
                    if G.get_edge_data(*edge)['weight'] >= theta:
                        e = tuple(sorted(edge))
                        if t > 1 and e in lambda_theta[t - 1].keys():
                            lambda_theta[t][e] = lambda_theta[t - 1][e] + 1
                        else:
                            lambda_theta[t][e] = 1
                    else:
                        G.remove_edge(*edge)
        UBR[1][t] = cal_S_rel(2 * len(lambda_theta[t].keys()) / k, 1, V_max, len(list_G), 1)
    return lambda_theta, UBR


def EEF(list_G_ori, query, theta, k, V_max, alpha):
    list_G = copy.deepcopy(list_G_ori)
    T_q = len(list_G)
    lambda_theta, UBR = bfs_lambda_theta(list_G, theta, k, V_max, source=query)

    # 追踪最优解
    max_score = -1.0
    best_results = []  # 列表存储 (score, G, interval)

    # 辅助集合用于快速检查重复的顶点集
    best_vertex_sets = []

    ubr_tuple = [(t, ubr) for t, ubr in UBR[1].items()]
    ubr_tuple = sorted(ubr_tuple, key=lambda x: x[1], reverse=True)

    for (t, ubr) in ubr_tuple:
        for d in range(1, t + 1):
            E_prime = [e for e, lam in lambda_theta[t].items() if lam >= d]
            UBR[d][t] = cal_S_rel(2 * len(E_prime) / k, d, V_max, T_q, alpha)

            if len(E_prime) >= k * (k + 1) / 2:
                C = local_k_core(list_G[t - 1].edge_subgraph(E_prime), query, k)

                # 计算分数
                S_rel = cal_S_rel(len(C.nodes), d, V_max, T_q, alpha)

                # 浮点数比较容差
                EPSILON = 1e-9

                if S_rel > max_score + EPSILON:
                    # 发现新高分，清空旧结果
                    max_score = S_rel
                    best_results = [(S_rel, C.copy(), (t - d + 1, t))]
                    best_vertex_sets = [frozenset(C.nodes())]

                elif abs(S_rel - max_score) < EPSILON:
                    # 发现同分结果，检查顶点集是否重复
                    current_nodes = frozenset(C.nodes())
                    if current_nodes not in best_vertex_sets:
                        best_results.append((S_rel, C.copy(), (t - d + 1, t)))
                        best_vertex_sets.append(current_nodes)

    if not best_results:
        print("EEF: No subgraph found.")
        return []

    print(f"EEF: Found {len(best_results)} optimal subgraph(s) with Score: {max_score:.4f}")
    # 返回所有最高分的子图对象列表
    return [res[1] for res in best_results]


# ======================== WCF (保持不变) =========================

# Establish Theta-threshold Table

def filter_theta(G, query, theta):
    G_temp = G.copy()
    for (u, v) in G_temp.edges:
        if G_temp[u][v]['weight'] * 10 < (theta + 1):
            G_temp.remove_edge(u, v)
    if query:
        for g in list(nx.connected_components(G_temp)):
            if query in g:
                G_temp = nx.subgraph(G_temp, g)
    return G_temp


def update_core_by_remove_theta(G, theta, df):
    origin_core = nx.core_number(G)
    G_temp = filter_theta(G, None, theta)
    filtered_core = nx.core_number(G_temp)
    for v, c in origin_core.items():
        if filtered_core[v] < c:
            for i in range(c - filtered_core[v]):
                df.loc[(v, c - i)] = round(theta * 0.1, 2)
    return G_temp


def theta_thres_table(G):
    k = k_max(G)[1]
    col_k = ['vertex'] + [i for i in range(1, k + 1)]
    init = dict.fromkeys(col_k)
    init['vertex'] = sorted(list(G.nodes))
    df_theta_thres = pd.DataFrame(init)
    df_theta_thres.set_index(['vertex'], inplace=True)
    G_prime = copy.deepcopy(G)
    for theta in range(11):
        G_prime = update_core_by_remove_theta(G_prime, theta, df_theta_thres)
    df_theta_thres = df_theta_thres.fillna(0)
    return df_theta_thres


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

    def remove_children(self, c_Node_id):
        self.children.remove(c_Node_id)

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
    k = k_max(G)[1]
    label = False
    for k_curr in range(1, k + 1):
        theta_tree_k = {}
        theta_tree_k['theta'] = {}
        theta_tree_k['node_id'] = {}
        ids = 0
        merged_ids = set()
        g = theta_thres_df.groupby(k_curr)
        theta_v = sorted(list(g.indices.keys()), reverse=True)
        for theta in theta_v:
            theta_tree_k['theta'][theta] = []
            node_v = g.get_group(theta).index.values.tolist()
            sub_G = nx.subgraph(G, node_v)
            temp_G = remove_theta(sub_G, theta)
            for C in list(nx.connected_components(temp_G)):
                merged = False
                X = Node(ids, list(C), theta)
                theta_tree_k['node_id'][ids] = X
                theta_tree_k['theta'][theta].append(ids)
                out_N = list(get_N_of_subgraph(nx.subgraph(G, C), G))
                visited = []
                for v in out_N:
                    if theta_thres_df.loc[v][k_curr] > theta and not merged:
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
                                        theta_tree_k['node_id'].pop(ids, 'merged')
                                        theta_tree_k['theta'][theta].remove(ids)
                                        merged_ids.add(ids)
                                    merged = True
                ids += 1
        WCF_index[k_curr] = theta_tree_k
    return WCF_index


# WCF-Index Based Query

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


def return_C1(G, wcf_index, query, theta, k):
    C_1 = None
    if k not in wcf_index.keys():
        return C_1
    tree = wcf_index[k]
    S = [node for node in tree['node_id'].keys() if is_root(tree, node, theta)]
    S = sorted(S, key=lambda x: tree['node_id'][x].theta)
    for root_id in S:
        C_vertices = tree['node_id'][root_id].get_subgraph_in_tree(tree['node_id'])[0]
        if query in C_vertices:
            G_k_max = nx.subgraph(G, C_vertices)
            C_1 = remove_theta(G_k_max, theta, query)
            C_1 = local_k_core(C_1, query, k)
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
    M = [len(L_c[1][t].nodes) for t in T_i]
    UBR = max(
        [cal_S_rel(mu, LCT(mu, M), V_max, T_q, alpha) for mu in M]
    )
    return UBR


# ======================== WCF Search Profiled (Modified) =========================

def WCF_search(list_G, WCF_indice, query, theta, k, V_max, alpha):

    T_q = len(list_G)
    all_size = [[0 for _ in range(T_q + 1)] for _ in range(T_q + 1)]
    L_c = [[nx.Graph() for _ in range(T_q + 1)] for _ in range(T_q + 1)]
    score = [[-1 for _ in range(T_q + 1)] for _ in range(T_q + 1)]
    anchored = []

    # 追踪最优解
    max_score = -1.0
    best_results = []  # [(score, G, interval), ...]
    best_vertex_sets = []  # [frozenset(nodes), ...]
    EPSILON = 1e-9

    # === Phase 1: Snapshot Initialization ===
    t0 = time.perf_counter()
    for t, wcf_index in enumerate(WCF_indice, start=1):
        C_1 = return_C1(list_G[t - 1], wcf_index, query, theta, k)
        if C_1 is None or len(C_1.nodes) == 0:
            anchored.append(t)
        else:
            L_c[1][t] = C_1
            score[1][t] = -len(C_1.nodes)

    # === Phase 2: Grouping & UBR Sorting ===
    t0 = time.perf_counter()
    T_elig = [t + 1 for t in range(len(list_G)) if t + 1 not in anchored]
    T_s = []
    for _, g in groupby(enumerate(T_elig), lambda x: x[0] - x[1]):
        T_s.append(list(map(itemgetter(1), g)))

    all_ubr = {tuple(T_i): UBR_wcf(T_i, L_c, V_max, T_q, alpha) for T_i in T_s}
    sorted_T_s = sorted(all_ubr.keys(), key=lambda x: all_ubr[x], reverse=True)
    # === Phase 3: DP & Intersection ===
    for T_i in sorted_T_s:
        t_prune = time.perf_counter()

        # 剪枝：如果当前时间段的理论上限 UBR 严格小于已找到的最高分，则跳过
        # 注意：如果 UBR == max_score，不能跳过，因为可能存在并列最优解
        if all_ubr[T_i] < max_score - EPSILON:
            continue

        for d in range(1, len(T_i) + 1):
            M = []
            for t in T_i:
                if d <= (t - T_i[0] + 1):
                    # --- Core Logic: Intersection ---
                    if d > 1:
                        t_inter = time.perf_counter()
                        inter = nx.intersection(L_c[d - 1][t - 1], L_c[d - 1][t])

                        t_core = time.perf_counter()
                        k_core_inter = local_k_core(inter, query, k)

                        if k_core_inter and len(k_core_inter.nodes) > 0:
                            L_c[d][t] = k_core_inter
                            all_size[d][t] = len(k_core_inter.nodes)

                    # --- Result Update & Tie Breaking ---
                    t_check = time.perf_counter()
                    if len(L_c[d][t].nodes) > 0:
                        S = cal_S_rel(len(L_c[d][t].nodes), d, V_max, T_q, alpha)
                        score[d][t] = S

                        # 更新全局最优解
                        if S > max_score + EPSILON:
                            max_score = S
                            best_results = [(S, L_c[d][t], (t - d + 1, t))]
                            best_vertex_sets = [frozenset(L_c[d][t].nodes())]

                        elif abs(S - max_score) < EPSILON:
                            # 处理并列情况
                            current_nodes = frozenset(L_c[d][t].nodes())
                            if current_nodes not in best_vertex_sets:
                                best_results.append((S, L_c[d][t], (t - d + 1, t)))
                                best_vertex_sets.append(current_nodes)

                        M.append(len(L_c[d][t].nodes))

            # --- LCT Pruning ---
            t_lct = time.perf_counter()
            if M:
                ubr = max([cal_S_rel(mu, LCT(mu, M) - 1 + d, V_max, T_q, alpha) for mu in M])
                # 只有当上限严格小于 max_score 时才终止内层循环
                if ubr < max_score - EPSILON:
                    break

    # 按照要求，返回所有得分最高且顶点集合不同的子图对象
    return [res[1] for res in best_results]