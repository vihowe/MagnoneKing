import multiprocessing
import threading
import time
import sys

sys.path.append('/home/vihowe/project/MagnoneKing/')

from PyQt5 import QtWidgets

from src.gui.view import MagnoneUi, MagnoneCtrl
from src.model.component import CpuGen, Node
from src.schedule.scheduling import Cluster, UserAgent

import logging


def update(win, p0):
    while True:
        cluster, load, avg_latency, instant_latency = p0.recv()
        logging.info(f'cluster:{cluster}, load: {load}, avg_latency: {avg_latency}, instant latency: {instant_latency}')
        win.set_load(load)
        win.set_cluster(cluster)
        win.set_avg_latency(avg_latency)
        win.set_instant_latency(instant_latency)


def simulate():
    app = QtWidgets.QApplication([])

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
    s0 = node_specification['desktop']
    s1 = node_specification['laptop']
    s2 = node_specification['pi']
    # for _ in range(2):
        # n = Node(node_id=node_id, cores=s0[0], mem=s0[1], core_gen=s0[2])
        # cluster.add_node(n)
        # node_id += 1
    for _ in range(2):
        n = Node(node_id=node_id, cores=s1[0], mem=s1[1], core_gen=s1[2])
        cluster.add_node(n)
        node_id += 1
    for _ in range(18):
        n = Node(node_id=node_id, cores=s2[0], mem=s2[1], core_gen=s2[2])
        cluster.add_node(n)
        node_id += 1

    config = {
        'slo': 1,
    }


    p0, p1 = multiprocessing.Pipe()
    win = MagnoneUi(cluster=cluster, comm_pipe=p0)
    win.show()
    magnoneCtrl = MagnoneCtrl(view=win)

    # view needs to communicate with the user agent to get the status of the cluster and current load

    config = {
        'slo': 3,
    }
    # user agent need to report the load and cluster status to the view
    user_agent = UserAgent(cluster, comm_pipe=p1, config=config)
    
    user_agent.start()

    T = threading.Thread(target=update, args=(win, p0))
    T.start()
    
    sys.exit(app.exec())
    


if __name__ == '__main__':
    multiprocessing.set_start_method('fork')
    logging.basicConfig(level=logging.INFO)
    simulate()
