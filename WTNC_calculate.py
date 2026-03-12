from collections import deque
import WTNC_cons

# 定义误差范围，用于判定浮点数是否相等
EPSILON = 1e-7


def update_best_results(best_results, score, component):
    """
    维护最高分结果列表：
    1. 遇到更高分 -> 清空列表，存入新结果
    2. 遇到同分 (在误差范围内) -> 检查顶点是否重复，不重复则追加
    """
    # 转换为 set 以便去重和比较，同时拷贝一份防止后续引用修改
    current_nodes_set = frozenset(component)

    # 1. 如果列表为空，直接添加
    if not best_results:
        best_results.append((score, current_nodes_set))
        return

    # 获取当前记录的最高分
    max_score = best_results[0][0]

    # 2. 如果新分数 显著大于 当前最高分 (超过误差范围)
    if score > max_score + EPSILON:
        best_results.clear()  # 清空之前的低分结果
        best_results.append((score, current_nodes_set))

    # 3. 如果新分数 近似等于 当前最高分 (在误差范围内)
    elif abs(score - max_score) <= EPSILON:
        # 检查是否已经存在完全相同的顶点集合
        exists = False
        for _, existing_nodes in best_results:
            if existing_nodes == current_nodes_set:
                exists = True
                break

        # 如果是新的连通分量，则追加
        if not exists:
            best_results.append((score, current_nodes_set))

    # 4. 如果分数显著小于最高分，则直接忽略


def calculate(node_list_all, inverted_index, theta, k, V_MAX, alpha, ts, te, query):
    # top_r 此时存储的是一个列表：[(score, frozenset(nodes)), ...]
    top_r = []

    # 初始化时间片上的边集合
    dur_temp = [[set() for _ in range(te - ts + 1)] for _ in range(te - ts + 1)]

    target_theta_idx = int(round(theta / 0.05))

    if query not in inverted_index:
        return []

    candidate_supernodes = inverted_index[query]
    seen_supernodes = set()

    # --- 第一阶段：基于倒排表的空间/时间扩展 (BFS_node) ---
    for info in candidate_supernodes:
        t = info['t']
        idx = info['node_list_idx']

        if t < ts or t > te:
            continue

        if info['k'] >= k and info['theta_idx'] >= target_theta_idx:
            if (t, idx) in seen_supernodes:
                continue
            seen_supernodes.add((t, idx))

            if t < len(node_list_all):
                current_node_list = node_list_all[t]
                # 在 BFS_node 内部直接更新 top_r
                BFS_node(current_node_list, idx, dur_temp, ts, te, theta, k, V_MAX, alpha, top_r, query)

    # --- 第二阶段：基于时间边的连通性检测 (Temporal Detect) ---
    d = te - ts - 1
    for i in range(d, 0, -1):
        for j in range(0, d - i + 1, 1):
            detect(k, V_MAX, alpha, dur_temp[j][i + j], d, top_r, i, query)
            # 传递边信息到下一层
            for edge_parent in dur_temp[j][i + j]:
                dur_temp[j][i + j - 1].add(edge_parent)
                dur_temp[j + 1][i + j].add(edge_parent)

    # 将 frozenset 转回 list 以便输出
    final_results = [(score, list(nodes)) for score, nodes in top_r]
    return final_results


def detect(k, V_MAX, alpha, edge, dur, top_r, x, query):
    adjacency_list = {}
    for source, target in edge:
        if source not in adjacency_list:
            adjacency_list[source] = []
        if target not in adjacency_list:
            adjacency_list[target] = []
        adjacency_list[source].append(target)
        adjacency_list[target].append(source)

    # 计算度数并进行 k-core 剪枝
    degree = {node: len(neighbors) for node, neighbors in adjacency_list.items()}
    low_degree_nodes = []

    # 初始筛选
    for node, deg in degree.items():
        if deg < k:
            low_degree_nodes.append(node)
            degree[node] = 0

    # 剥洋葱过程
    idx = 0
    while idx < len(low_degree_nodes):
        vertex = low_degree_nodes[idx]
        idx += 1
        for neighbors_ver in adjacency_list.get(vertex, []):
            if degree[neighbors_ver] == 0:
                continue
            degree[neighbors_ver] -= 1
            if degree[neighbors_ver] < k:
                low_degree_nodes.append(neighbors_ver)
                degree[neighbors_ver] = 0

    # 对剩下的节点划分连通分量
    visited = set()
    for node_com in degree:
        if degree[node_com] != 0 and node_com not in visited:
            bfs(node_com, V_MAX, adjacency_list, alpha, dur, degree, top_r, x, query, visited)


def bfs(start_node, V_MAX, adjacency_list, alpha, dur, degree, top_r, x, query, visited_global):
    component = []
    queue = deque([start_node])
    visited_global.add(start_node)
    component.append(start_node)

    # 这里为了避免重复计算 degree，我们假设进入这里的都是满足 k-core 的
    # 但为了标记 component，我们需要局部遍历

    # 注意：degree 数组在 detect 里已经标记了是否被剔除(0)，这里不需要再置0，只需要根据 connectivity 遍历
    # 但为了防止后续重复遍历，可以在 visited_global 记录

    current_visited = {start_node}

    while queue:
        u = queue.popleft()
        for v in adjacency_list.get(u, []):
            if degree[v] != 0 and v not in current_visited:
                current_visited.add(v)
                visited_global.add(v)
                component.append(v)
                queue.append(v)

    if query not in component:
        return

    score = WTNC_cons.cal_S_rel(len(component), x + 1, V_MAX, dur + 1, alpha)
    update_best_results(top_r, score, component)


def BFS_node(node_list, node_idx, dur_temp, ts, te, theta, k, V_MAX, alpha, top_r, query):
    visit = [0 for _ in range(len(node_list))]
    compoent = set()  # 存储原始顶点
    queue = deque([node_idx])
    visit[node_idx] = 1

    while queue:
        u = queue.popleft()

        # 收集顶点
        for ver in node_list[u].vertex:
            compoent.add(ver)

        # 收集时间边
        for edge in node_list[u].edge_side:
            if len(edge) < 2: continue
            start_time = max(ts, edge[0])
            end_time = min(te, edge[1])
            if end_time < start_time: continue

            for i in range(2, len(edge)):
                source, target = edge[i]
                edge_weight = node_list[u].theta
                if edge_weight >= theta:
                    dur_temp[start_time - ts][end_time - ts].add(tuple(sorted((source, target))))

        # 收集空间邻居
        for neighbor, weight in node_list[u].edge:
            if visit[neighbor] > 0: continue
            if node_list[neighbor].theta >= theta and node_list[neighbor].k >= k and weight >= theta:
                visit[neighbor] = 1
                queue.append(neighbor)

    if query not in compoent:
        return

    score_com = WTNC_cons.cal_S_rel(len(compoent), 1, V_MAX, te - ts, alpha)

    # 立即尝试更新结果列表
    update_best_results(top_r, score_com, compoent)