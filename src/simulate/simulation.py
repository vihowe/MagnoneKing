import multiprocessing
import threading
import time
import sys

sys.path.append('/home/vihowe/project/MagnoneKing/')

from PyQt5 import QtWidgets

from src.gui.view import MagnoneUi, MagnoneCtrl
from src.model.component import CpuGen, Node
from src.schedule.scheduling import Cluster, UserAgent


def update(win, p0):
    while True:
        cluster, load = p0.recv()
        win.set_load(load)
        win.set_cluster(cluster)

def simulate():
    app = QtWidgets.QApplication([])

    cluster = Cluster()

    node_specification = {
        "desktop": (12, 8192, CpuGen.D,),
        "laptop": (10, 4096, CpuGen.C,),
        "phone": (8, 2048, CpuGen.B,),
        "pi": (4, 2048, CpuGen.A,),
    }
    node_id = 1
    for v in node_specification.values():
        for _ in range(5):
            n = Node(node_id=node_id, cores=v[0], mem=v[1], core_gen=v[2])
            cluster.add_node(n)
            node_id += 1
    p0, p1 = multiprocessing.Pipe()
    win = MagnoneUi(cluster=cluster, comm_pipe=p0)
    win.show()
    magnoneCtrl = MagnoneCtrl(view=win)

    # view needs to communicate with the user agent to get the status of the cluster and current load

    config = {
        'slo': 3,
    }
    user_agent = UserAgent(cluster, comm_pipe=p1, config=config)
    user_agent.start()

    T = threading.Thread(target=update, args=(win, p0))
    T.start()
    sys.exit(app.exec())


if __name__ == '__main__':
    simulate()


