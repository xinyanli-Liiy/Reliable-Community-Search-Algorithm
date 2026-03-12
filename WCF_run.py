import click
import time
import WCF_CRC
from joblib import Parallel, delayed
import networkx as nx


def save_results_to_file(file_name, results, query_time, index_time):
    """
    保存结果到文件。
    由于新的接口只返回最优子图列表 [G1, G2, ...],
    不再包含具体的得分和时间区间信息（这些信息已在控制台打印）。
    """
    with open('./Output/' + file_name, 'w') as f:
        # 写入时间性能信息
        if index_time is not None:
            f.write(f'Index construction time: {index_time:.6f} s\n')
        f.write(f'Running time of query: {query_time:.6f} s\n')

        if not results:
            f.write('\nNo subgraph found satisfying the conditions.\n')
            print(f'No results to save for {file_name}.')
            return

        f.write(f'\nFound {len(results)} optimal subgraph(s):\n')
        f.write('=' * 40 + '\n')

        for idx, G_opt in enumerate(results, 1):
            f.write(f'Option {idx}:\n')
            # 因为不再返回分数和区间，这里只记录节点和边数作为基本信息
            f.write(f'  Vertices Size: {len(G_opt.nodes)}\n')
            f.write(f'  Edges Size:    {len(G_opt.edges)}\n')
            f.write(f'  Nodes List:    {list(G_opt.nodes)}\n')
            f.write('-' * 40 + '\n')

    print('Optimal results saved to:', file_name)


@click.command()
@click.option('--dataset', prompt='Dataset name(str)', help='The name of the dataset')
@click.option('--theta', prompt='Theta(float)', help='The value of the parameter Theta')
@click.option('--k', prompt='K(int)', help='The value of the parameter K')
@click.option('--query', prompt='Query(str)', help='The value of the Query vertex')
@click.option('--alpha', prompt='Alpha(float)', help='The value of the parameter Alpha')
@click.option('--start_time', prompt='T_s(int)', help='The start timestamp(included, start from 0)')
@click.option('--end_time', prompt='T_e(int)', help='The end timestamp(excluded)')
def query(dataset, theta, k, query, alpha, start_time, end_time):
    theta = float(theta)
    k = int(k)
    ts = int(start_time)
    te = int(end_time)
    alpha = float(alpha)

    # 1. 加载数据
    list_G = WCF_CRC.get_list_G(dataset, ts, te)
    V_MAX = WCF_CRC.get_V_max(list_G, k)

    # 2. 执行 EEF 算法
    print(f"\n--- Running EEF on {dataset} ---")
    start = time.perf_counter()
    # 注意：EEF 现已移除 r 参数
    top_communities_eef = WCF_CRC.EEF(list_G, query, theta, k, V_MAX, alpha)
    end = time.perf_counter()
    EEF_time = end - start

    file_name_eef = '{}-{}-{}-{}-{}_EEF.txt'.format(dataset, theta, k, query, alpha)
    save_results_to_file(file_name_eef, top_communities_eef, EEF_time, None)

    # 3. 构建索引 (WCF Index)
    print(f"\n--- Building WCF Index ---")
    start1 = time.perf_counter()
    theta_thres_all = Parallel(n_jobs=-1)(delayed(WCF_CRC.theta_thres_table)(g) for g in list_G)
    wcf_indices = Parallel(n_jobs=-1)(delayed(WCF_CRC.theta_tree)(theta_thres_all[i], g) for i, g in enumerate(list_G))
    end1 = time.perf_counter()
    index_time = end1 - start1

    # 4. 执行 WCF 搜索
    print(f"\n--- Running WCF Search on {dataset} ---")
    start2 = time.perf_counter()
    # 注意：函数名已更新为 WCF_search_profiled，且移除了 r 参数
    top_communities_wcf = WCF_CRC.WCF_search(list_G, wcf_indices, query, theta, k, V_MAX, alpha)
    end2 = time.perf_counter()
    WCF_time = end2 - start2

    file_name_wcf = '{}-{}-{}-{}-{}_WCF.txt'.format(dataset, theta, k, query, alpha)
    save_results_to_file(file_name_wcf, top_communities_wcf, WCF_time, index_time)


if __name__ == '__main__':
    query()