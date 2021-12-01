r"""
read the profiled data and generate the affinity priority of each type of task
and the appropriate amount of each task with respect to each node
"""
import os
from collections import defaultdict

import pandas as pd
import numpy as np
import sys
import random


sys.path.append('/home/vihowe/project/MagnoneKing/')
from src.model.component import CpuGen, Node, TaskType, Cluster


def get_fit_res(data):
    """Get the appropriate amount resource whilst achieving the best performance"""
    weight = 50
    threshold = 0.8
    data['total_time'] = data['running time'] * weight + data['overhead']

    m = data[(data['cpu_quota'] == 100000) | (data['cpu_quota'] == 1)]['mem_quota'].tolist()
    d = data[(data['cpu_quota'] == 100000) | (data['cpu_quota'] == 1)]['total_time'].tolist()
    t = data[(data['cpu_quota'] == 100000) | (data['cpu_quota'] == 1)]['running time'].tolist()
    for i in range(len(d)):
        if d[i+1] / d[i] > threshold:
            mem = m[i]
            running_time = t[i]
            break

    return 1, mem, running_time


def get_res_time(cluster: Cluster):
    """return the resource-time table for each (task, node) pair

    Return:
        res[task_id]: a list of (node_id, cpu_quota, mem_quota, running_time) in descending order of affinity priority
    """
    task_dics = [defaultdict(tuple) for i in range(4)]

    task_types = []
    for _, task_type in TaskType.__members__.items():
        task_types.append(task_type)
    
    nodes = cluster.nodes

    for c_node in nodes:
        cpu_gen: CpuGen
        cpu_gen = c_node.core_gen
        data = pd.read_csv(os.path.join('/home/vihowe/project/MagnoneKing/data', str(cpu_gen.value)+'.csv'))
        for task in task_types:
            task_data = data[task.value == data['task']]
            r_t = get_fit_res(task_data)
            task_dics[task.value][c_node.node_id] = (c_node.node_id, *r_t)

    res = {}
    for task in task_types:
        reverse = True if task == TaskType.A or task == TaskType.C else False
        # reverse = False
        vs = list(task_dics[task.value].values())
        vs = sorted(vs, key=lambda x: x[-1], reverse=reverse)
        # random.shuffle(vs)


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

    cluster = Cluster()

    node_specification = {
        "desktop": (8, 8192, CpuGen.A,),
        "laptop": (4, 4096, CpuGen.B,),
        "pi": (2, 2048, CpuGen.C,),
    }
    node_id = 1
    # for v in node_specification.values():
    #     for _ in range(2):
    #         n = Node(node_id=node_id, cores=v[0], mem=v[1], core_gen=v[2])
    #         cluster.add_node(n)
    #         node_id += 1
    s1 = node_specification['laptop']
    s2 = node_specification['pi']
    for _ in range(2):
        n = Node(node_id=node_id, cores=s1[0], mem=s1[1], core_gen=s1[2])
        cluster.add_node(n)
        node_id += 1
    for _ in range(6):
        n = Node(node_id=node_id, cores=s2[0], mem=s2[1], core_gen=s2[2])
        cluster.add_node(n)
        node_id += 1

    config = {
        'slo': 1,
    }
    
    res = get_res_time(cluster)
    print(res)




