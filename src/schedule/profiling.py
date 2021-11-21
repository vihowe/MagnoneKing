r"""
read the profiled data and generate the affinity priority of each type of task
and the appropriate amount of each task with respect to each node
"""
import os
from collections import defaultdict

import pandas as pd
import numpy as np

from src.model.component import TaskType


def get_fit_res(data):
    """Get the appropriate amount resource whilst achieving the best performance"""
    weight = 50
    threshold = 0.8
    data['total_time'] = data['running time'] * weight + data['overhead']
    for mem_quota in np.arange(40, 240, 20):
        m = data[(data['cpu_quota'] == 100000) | (data['cpu_quota'] == 1)]['mem_quota'].tolist()
        d = data[(data['cpu_quota'] == 100000) | (data['cpu_quota'] == 1)]['total_time'].tolist()
        t = data[(data['cpu_quota'] == 100000) | (data['cpu_quota'] == 1)]['running time'].tolist()
        for i in range(len(d)):
            if d[i+1] / d[i] > threshold:
                mem = m[i]
                running_time = t[i]
                break

    return 1, mem, running_time


def get_res_time(file_names):
    """return the resource-time table for each (task, node) pair

    Args:
        file_names: actually representing the node's id

    Return:
        res[task_id]: a list of (node_id, cpu_quota, mem_quota, running_time) in descending order of affinity priority
    """
    task_dics = [defaultdict(tuple) for i in range(4)]

    task_types = []
    for _, task_type in TaskType.__members__.items():
        task_types.append(task_type)

    for node_id in file_names:
        data = pd.read_csv(os.path.join('../../data', str(node_id)+'.csv'))
        for task in task_types:
            task_data = data[task.value == data['task']]
            r_t = get_fit_res(task_data)
            task_dics[task.value][node_id] = (node_id, *r_t)

    res = {}
    for task in task_types:
        vs = list(task_dics[task.value].values())
        vs = sorted(vs, key=lambda x: x[-1])
        res[task] = vs

    return res


# def get_affinity_priority(task_type: TaskType, cluster: Cluster):
#     task_data = []  # each item contains the appropriate amount resource in one node and its expected running time
#     for c_node in cluster.nodes:
#         c_node: Node
#         data_path = os.path.join('./data', f'{c_node.node_id}.csv')
#         # TODO find the data of this task_type in the c_node
#         t_node = pd.read_csv(data_path, header=None)
#         task_data.append(t_node)


if __name__ == '__main__':
    file_names = [0, 1, 2]
    res = get_res_time(file_names)
    print(res)




