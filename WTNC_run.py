import click
import time
from joblib import Parallel, delayed
from collections import defaultdict
import WTNC_cons
import WCF_CRC
import WTNC_calculate


def save_best_results_to_file(file_name, best_results, query_time, index_time):
    with open('./Output/' + file_name, 'w') as f:
        f.write(f'Index construction time: {index_time}\n')
        f.write(f'Running time of WCF query: {query_time}\n')

        if best_results and len(best_results) > 0:
            # 这里的 best_results 列表里的每一个都是最高分（或近似最高分）
            max_score = best_results[0][0]
            f.write(f'Found {len(best_results)} result(s) with Highest Score: {max_score}\n')
            f.write('=' * 40 + '\n')

            for idx, (score, nodes) in enumerate(best_results, 1):
                f.write(f'Result #{idx}:\n')
                f.write(f'Score: {score}\n')
                # 排序后输出以便阅读
                f.write(f'Nodes ({len(nodes)}): {sorted(list(nodes))}\n')
                f.write('-' * 40 + '\n')
        else:
            f.write('No results found.\n')

    print('CRC output saved to:', file_name)


@click.command()
@click.option('--dataset', prompt='Dataset name(str)', help='The name of the dataset')
@click.option('--theta', prompt='Theta(float)', help='The value of the parameter Theta')
@click.option('--k', prompt='K(int)', help='The value of the parameter K')
@click.option('--query', prompt='Query(str)', help='The value of the Query vertex')
@click.option('--alpha', prompt='Alpha(float)', help='The value of the parameter Alpha')
@click.option('--start_time', prompt='T_s(int)', help='The start timestamp(included, start from 0)')
@click.option('--end_time', prompt='T_e(int)', help='The end timestamp(excluded)')
def query(dataset, theta, k, alpha, start_time, end_time, query):
    theta = float(theta)
    k = int(k)
    ts = int(start_time)
    te = int(end_time)
    alpha = float(alpha)

    list_g = WCF_CRC.get_list_G(dataset, ts, te)
    V_MAX = WCF_CRC.get_V_max(list_g, k)

    # === Index Construction Phase ===
    start = time.perf_counter()

    node_list_all = Parallel(n_jobs=-1)(delayed(WTNC_cons.theta_thres_table)(list_g[i], i) for i in range(len(list_g)))
    WTNC_cons.id_distribution(node_list_all)

    inverted_index = defaultdict(list)
    for i in range(len(node_list_all)):
        WTNC_cons.edge_decompose(list_g[i], node_list_all[i], i, inverted_index)

    for i in range(len(node_list_all)):
        t = node_list_all[i][len(node_list_all[i]) - 1].id
        WTNC_cons.edge_duration(list_g[t], node_list_all[i], list_g, t)

    end = time.perf_counter()
    index_time = end - start

    # === Query Phase ===
    start2 = time.perf_counter()

    # 获取所有并列第一的结果
    best_results = WTNC_calculate.calculate(node_list_all, inverted_index, theta, k, V_MAX, alpha, ts, te, query)

    end2 = time.perf_counter()
    query_time = end2 - start2

    file_name_WTNC = '{}-{}-{}-{}_WTNC'.format(dataset, theta, k, alpha)
    save_best_results_to_file(file_name_WTNC, best_results, query_time, index_time)


if __name__ == '__main__':
    query()