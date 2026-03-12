from collections import deque
from pathlib import Path
from typing import Any
import networkx as nx
import pandas as pd
from collections import deque, defaultdict


def get_list_G(dataset, ts, te):  # 读取图（主函数)
    graph_path = './Data/' + dataset + '/'
    graph_files = list(Path(graph_path).glob('*'))[ts:te]
    list_G: list[Any] = [nx.read_gml(g) for g in graph_files]
    return list_G


def k_max(G):
    return sorted(list(nx.core_number(G).items()), key=lambda x: x[1], reverse=True)[0]


def cal_S_rel(V_c, T_c, V_max, T_q, alpha):  # 得分函数#已
    aa = (1 + alpha * alpha) * (V_c / V_max * T_c / T_q) / (alpha * alpha * V_c / V_max + T_c / T_q)
    return aa


class ConnectionNode:
    def __init__(self):
        self.vertex = []  # 存储顶点的列表
        self.k = 0  # k值
        self.theta = 0.0  # θ值 (changed to float)
        self.edge = []
        self.start_time = 0
        self.end_time = 0
        self.id = 0
        self.edge_side = [[]]

    def add_node(self, vertex):
        if vertex not in self.vertex:  # 确保不重复添加同一个顶点
            self.vertex.append(vertex)

    def set_k(self, k):
        self.k = k

    def set_theta(self, theta):
        self.theta = theta

    def add_edge(self, edge):
        if edge not in self.edge:  # 确保不重复添加同一条边
            self.edge.append(edge)

    def set_start_time(self, start_time):
        self.start_time = start_time

    def set_end_time(self, end_time):
        self.end_time = end_time

    def set_id(self, id):
        self.id = id


def id_distribution(node_list_all):
    id = [[]]
    for i in range(len(node_list_all)):
        for j in range(len(node_list_all[i]) - 1):
            flag = 0
            for k in range(len(id) - 1):
                if node_list_all[i][j].vertex == node_list_all[id[k][0]][id[k][1]].vertex:
                    if node_list_all[i][j].theta == node_list_all[id[k][0]][id[k][1]].theta:
                        if node_list_all[i][j].k == node_list_all[id[k][0]][id[k][1]].k:
                            node_list_all[i][j].id = k
                            id[k].append(i)
                            id[k].append(j)
                            flag = 1
                            break
            if flag == 0:
                temp = len(id) - 1
                id.extend([] for _ in range(1))
                id[temp].append(i)
                id[temp].append(j)
                node_list_all[i][j].id = temp
    new_id = len(id)
    for i in range(len(id)):
        timi = [0 for _ in range(len(node_list_all) + 1)]
        group = []
        for j in range(0, len(id[i]) - 1, 2):
            timi[node_list_all[id[i][j]][id[i][j + 1]].end_time] = 1
        if len(timi) == 1:
            for k in range(0, len(id[i]), 2):
                node_list_all[k][k + 1].start_time = 0
                node_list_all[k][k + 1].end_time = 0
        if len(timi) > 1:
            r = timi[0]
            for n in range(1, len(node_list_all), 1):
                l = r
                r = timi[n]
                if n == 1:
                    if l == 1 and r == 1:
                        group.append(n - 1)
                if n > 1:
                    if l == 1 and r == 1:
                        group.append(n)
                if l == 0 and r == 1:
                    group.append(n)
                if l == 1 and r == 0:
                    group.append(n - 1)
            for s in range(0, len(group) - 1, 2):
                temp_id = i
                if group[s] != group[0]:
                    temp_id = new_id
                    new_id = new_id + 1
                for p in range(0, len(id[i]) - 1, 2):
                    if node_list_all[id[i][p]][id[i][p + 1]].start_time >= group[s] and node_list_all[id[i][p]][id[i][p + 1]].end_time <= group[s + 1]:
                        node_list_all[id[i][p]][id[i][p + 1]].start_time = group[s]
                        node_list_all[id[i][p]][id[i][p + 1]].end_time = group[s + 1]
                        node_list_all[id[i][p]][id[i][p + 1]].id = temp_id


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


def theta_thres_table(G, t):
    k = k_max(G)[1]
    node_list = []
    df_theta_thres = pd.DataFrame(index=sorted(G.nodes()),
                      columns=range(1, k + 1))

    for theta in [round(x * 0.01, 2) for x in range(1, 101)]:  # 0.01到1.00
        G_filtered = filter_theta(G, theta)
        core_numbers = nx.core_number(G_filtered)
        for v, c in core_numbers.items():
            for k_val in range(1, c + 1):
                if pd.isna(df_theta_thres.at[v, k_val]) or theta > df_theta_thres.at[v, k_val]:
                    df_theta_thres.at[v, k_val] = theta
    df_theta_thres = df_theta_thres.round(2).fillna(0)
    # 将顶点分类
    sc = [[[[] for _ in range(0)] for _ in range(0)] for _ in range(21)]  # 0.0-1.0 in 0.05 steps

    for vertex in df_theta_thres.index:
        theta_series_temp = df_theta_thres.loc[vertex]
        theta_series = []
        theta_after = -1.0
        for k_str, theta in reversed(list(theta_series_temp.items())):
            if theta_after == -1.0:
                theta_series.append((k_str, theta))
                theta_after = theta
            elif theta > theta_after:
                theta_series.append((k_str, theta))
                theta_after = theta

        for k_str, theta in theta_series:
            if theta != 0.0:
                theta_index = int(round(theta / 0.05, 5))
                if len(sc[theta_index]) <= k_str:
                    current_length = len(sc[theta_index])
                    additional_length = k_str + 1 - current_length
                    if additional_length > 0:
                        sc[theta_index].extend([[] for _ in range(additional_length)])
                sc[theta_index][k_str].append(vertex)
    for i in range(len(sc) - 1, -1, -1):
        theta_val = round(i * 0.05, 5)  # Get the actual theta value
        for j in range(len(sc[i]) - 1, -1, -1):
            if len(sc[i][j]) == 0:
                continue
            com = {node: -1 for node in G.nodes()}
            c = 1
            for v in sc[i][j]:
                queue = deque([v])
                if com[v] > 0:
                    continue
                com[v] = c
                # BFS
                while queue:
                    u = queue.popleft()
                    for neighbor in G.neighbors(u):
                        if com[neighbor] == -1:
                            if G[u][neighbor]['weight'] >= theta_val:
                                com[neighbor] = 0
                                theta_nei = df_theta_thres.loc[neighbor]
                                for neighbor_k, neighbor_theta in theta_nei.items():
                                    if neighbor_k >= j and neighbor_theta >= theta_val:
                                        queue.append(neighbor)
                                        com[neighbor] = c
                                        break
                c = c + 1

            # Create nodes
            temp_list = [ConnectionNode() for _ in range(c - 1)]
            for v in sc[i][j]:
                temp_list[com[v] - 1].add_node(v)
                temp_list[com[v] - 1].set_theta(theta_val)  # Store actual theta value
                temp_list[com[v] - 1].set_k(j)

            for temp in temp_list:
                temp.start_time = t
                temp.end_time = t
                node_list.append(temp)

    root = ConnectionNode()
    root.id = t
    root.theta = 0.0
    root.k = 0.0
    node_list.append(root)
    return node_list


def edge_decompose(G, node_list, t, inverted_index):  # 注意：参数 WCAG 替换为 inverted_index
    max_node = 0
    for i in G.nodes:
        if int(i) > max_node:
            max_node = int(i)

    # 辅助数组，用于快速查找顶点所在的 node_list 索引
    super_id = [[] for _ in range(max_node + 1)]
    for i in range(len(node_list) - 1):
        for vertex in node_list[i].vertex:
            super_id[int(vertex)].append(i)

    # se 存储边信息，用于构建超节点间的连接关系
    # 0.00-0.95 in 0.05 steps (20 intervals)
    se = [[[[] for _ in range(0)] for _ in range(0)] for _ in range(20)]

    for i in range(len(node_list)):
        visited = [0 for _ in range(len(node_list))]
        for vertex in node_list[i].vertex:
            for neighbor in G.neighbors(vertex):
                if G[vertex][neighbor]['weight'] < node_list[i].theta:
                    continue
                for id in super_id[int(neighbor)]:
                    if visited[id] > G[vertex][neighbor]['weight']:
                        continue
                    visited[id] = G[vertex][neighbor]['weight']
        for j in range(len(visited)):
            if visited[j] > 0:
                min_k = int(min(node_list[i].k, node_list[j].k))
                min_theta = min(node_list[i].theta, node_list[j].theta, visited[j])
                theta_index = int(min_theta // 0.05)
                theta_index = min(theta_index, 19)

                if len(se[theta_index]) <= min_k:
                    current_length = len(se[theta_index])
                    additional_length = min_k + 1 - current_length
                    if additional_length > 0:
                        se[theta_index].extend([[] for _ in range(additional_length)])
                se[theta_index][min_k].append([i, j, visited[j]])

    # 填充 se 数组的空隙
    for i in range(len(se) - 2, -1, -1):
        if len(se[i]) < len(se[i + 1]):
            current_length = len(se[i])
            additional_length = len(se[i + 1]) + 1 - current_length
            if additional_length > 0:
                se[i].extend([[] for _ in range(additional_length)])

    # === 核心修改部分：构建倒排索引 ===
    # 遍历 Theta 层级 (i) 和 K 层级 (j)
    for i in range(len(se) - 1, -1, -1):
        theta_val = i * 0.05
        for j in range(len(se[i]) - 1, 0, -1):

            # 1. 并在查集/BFS 初始化，寻找连通分量
            com = [0 for _ in range(len(node_list))]
            c = 1
            for k in range(len(node_list) - 1):
                if node_list[k].theta >= theta_val and node_list[k].k >= j and com[k] == 0:
                    queue = deque([k])
                    com[k] = c
                    while queue:
                        u = queue.popleft()
                        for neighbor, weight in node_list[u].edge:
                            if com[neighbor] > 0: continue
                            if weight < theta_val: continue
                            if node_list[neighbor].theta >= theta_val and node_list[neighbor].k >= j:
                                com[neighbor] = c
                                queue.append(neighbor)
                    c = c + 1

            # 2. 并查集路径压缩 (pre 数组)
            pre = list(range(c))
            for u, v, weight in se[i][j]:
                ucom = com[u]
                vcom = com[v]
                if ucom == 0 or vcom == 0: continue  # 跳过未被标记的节点

                while ucom != pre[ucom]: ucom = pre[ucom]
                while vcom != pre[vcom]: vcom = pre[vcom]

                if ucom != vcom:
                    node_list[u].edge.append((v, weight))
                    node_list[v].edge.append((u, weight))
                    # 简单的合并策略，这里可以优化为按秩合并
                    pre[vcom] = ucom

            # 3. 收集连通分量 (Component)
            dis = defaultdict(list)
            for q in range(len(com)):
                if com[q] != 0:
                    root = com[q]
                    while root != pre[root]:
                        root = pre[root]
                    # 更新路径压缩（可选）
                    # pre[com[q]] = root
                    dis[root].append(q)  # 将 node_list 的索引 q 加入对应的连通分量 root

            # 4. === 构建倒排表 (Inverted Index) ===
            # dis 里的每一个 item (root, w) 都是一个“超节点”集合
            for root_id, w_indices in dis.items():
                if not w_indices: continue

                # 提取该超节点的属性
                # 注意：这里逻辑沿用你之前的 w[len(w) - 1] 作为代表元
                representative_idx = w_indices[-1]

                # 计算该超节点包含的所有原始顶点 (去重)
                unique_elements = set()
                for idx in w_indices:
                    for v in node_list[idx].vertex:
                        unique_elements.add(v)

                size = len(unique_elements)

                # 记录超节点信息对象
                super_node_info = {
                    't': t,
                    'node_list_idx': representative_idx,  # 对应 node_list 中的下标
                    'size': size,
                    'theta_idx': i,  # 记录具体的 Theta 索引 (0-19)
                    'k': j  # 记录具体的 K 值
                }

                # 将此超节点信息“挂”到它包含的每一个原始顶点下
                for v in unique_elements:
                    inverted_index[v].append(super_node_info)


def edge_duration(G, node_list, list_g, ti):
    for i in range(len(node_list)):
        seen_edges = set()
        for source in node_list[i].vertex:
            for t in list(G.edges(source)):
                target = t[1]
                # Directly compare the precise weight
                if G.get_edge_data(source, target)['weight'] >= node_list[i].theta:
                    edge = tuple(sorted([source, target]))
                    if edge not in seen_edges:
                        seen_edges.add(edge)
                        st = ca_start(list_g, ti, source, target, node_list[i].theta)
                        se = ca_end(list_g, ti, source, target, node_list[i].theta)
                        temp2 = 0
                        for array in node_list[i].edge_side:
                            if len(array) < 2:
                                continue
                            if array[0] == st and array[1] == se:
                                temp2 = 1
                                if target in node_list[i].vertex:
                                    array.append((source, target))
                                else:
                                    array.append((source, target))
                                break
                        if temp2 == 0:
                            new_array = []
                            new_array.append(st)
                            new_array.append(se)
                            if target in node_list[i].vertex:
                                sorted_edge = sorted([source, target])
                                new_array.append((sorted_edge[0], sorted_edge[1]))  # 小节点在前
                            else:
                                # 外部边：排序后存储
                                sorted_edge = sorted([source, target])
                                new_array.append((sorted_edge[0], sorted_edge[1]))  # 小节点在前
                            node_list[i].edge_side.append(new_array)


def ca_start(list_g, ti, source, target, weight):
    st = ti
    if (source, target) in list_g[ti].edges:
        if list_g[ti][source][target]['weight'] >= weight:
            if ti == 0:
                return st
            else:
                st = ca_start(list_g, ti - 1, source, target, weight)
                return st
    return st + 1


def ca_end(list_g, ti, source, target, weight):
    se = ti
    if (source, target) in list_g[ti].edges:
        if list_g[ti].get_edge_data(source, target)['weight'] >= weight:
            if ti == len(list_g) - 1:
                return se
            else:
                se = ca_end(list_g, ti + 1, source, target, weight)
                return se
    return se - 1
