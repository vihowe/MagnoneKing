import multiprocessing
import sys
import functools
import threading
import time

import PyQt5.QtWidgets as QtWidgets
import networkx as nx
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import PyQt5.QtGui as QtGui
import random
import collections

from matplotlib.backends.qt_compat import QtCore, QtWidgets

from src.model.component import CpuGen, Node
from src.schedule.scheduling import Cluster

if QtCore.qVersion() >= "5.":
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

from src.model import component


class MagnoneUi(QtWidgets.QMainWindow):
    """Magnone's View (GUI)
    Attributes:
        _cluster: the cluster consists of nodes its is responsible to show
        comm_pipe: the communication pipe with the user agent
    """

    def __init__(self, cluster: Cluster, comm_pipe: multiprocessing.Pipe):
        """View initializer"""
        super(MagnoneUi, self).__init__()
        self._load = 0.1
        self._cluster = cluster
        self.comm_pipe = comm_pipe
        self.setWindowTitle('磁探测')
        self.setFixedSize(1200, 1000)
        self._centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self._centralWidget)
        self.generalLayout = QtWidgets.QVBoxLayout()
        self._centralWidget.setLayout(self.generalLayout)
        self._createMenu()
        self._createCtrlButton()
        self._createLoadPanel()
        self._createNodesPanel()

    @property
    def cluster(self):
        return self._cluster

    @cluster.setter
    def cluster(self, value):
        self.cluster = value

    def _createMenu(self):
        self.menu = self.menuBar().addMenu("&Menu")
        self.menu.addAction("&Exit", self.close)

    def _createCtrlButton(self):
        self.buttonLayout = QtWidgets.QHBoxLayout()
        self.buttons = {'start': QtWidgets.QPushButton('Start'), 'stop': QtWidgets.QPushButton('Stop'),
                        'reset': QtWidgets.QPushButton('Reset')}
        for name, button in self.buttons.items():
            self.buttonLayout.addWidget(button)
        self.generalLayout.addLayout(self.buttonLayout)

    def _createLoadPanel(self, init_pic='img/load0.png'):
        """The panel shows the status of workload"""
        # self.loadPanel = QtWidgets.QLabel()
        # self.loadPanel.setFixedSize(1200, 400)
        # pixmap = QtGui.QPixmap(init_pic)
        # self.loadPanel.setPixmap(pixmap)
        # self.generalLayout.addWidget(self.loadPanel)
        load_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        self.generalLayout.addWidget(load_canvas)
        self._load_ax = load_canvas.figure.subplots()
        x = np.linspace(0, 100, 100)
        self._cur_load = collections.deque(np.zeros_like(x), maxlen=100)
        self._load_ax.set_ylim(0, 100)
        self._load_ax.set_xlim(0, 100)
        self._load_ax.set_xticks([])
        self._load_line, = self._load_ax.plot(x, self._cur_load)
        self._load_timer = load_canvas.new_timer(900)
        self._load_timer.add_callback(self._updateLoadPanel)
        # self._load_timer.start()

    def _updateLoadPanel(self):
        x = np.linspace(0, 100, 100)
        self._cur_load.append(self._load)
        self._load_line.set_data(x, self._cur_load)
        self._load_line.figure.canvas.draw()

    def _createNodesPanel(self):
        nodes_canvas = FigureCanvas(Figure(figsize=(10, 8)))
        self.generalLayout.addWidget(nodes_canvas)
        self._nodes_ax = nodes_canvas.figure.subplots()

        self._G = nx.Graph()
        self._nodes = []
        for c_node in self._cluster.nodes:
            self._nodes.append(c_node)

        # self._edges = []
        # for i in range(20):
        #     node_i = random.choice(self._nodes)
        #     node_j = random.choice(self._nodes)
        #     if node_i != node_j:
        #         self._edges.append((node_i, node_j))
        self._G.add_nodes_from(self._nodes)
        # self._G.add_edges_from(self._edges)
        self._nodes_pos = nx.spring_layout(self._G, k=1)
        color_map = ['#33A6CC' if c_node.activated else 'gray'
                     for c_node in self._nodes]
        nx.draw(self._G, ax=self._nodes_ax, pos=self._nodes_pos,
                node_color=color_map)
        self._nodes_timer = nodes_canvas.new_timer(900)
        self._nodes_timer.add_callback(self._updateNodesPanel)

    def _updateNodesPanel(self):
        self._nodes_ax.clear()
        self._nodes = []
        for c_node in self.cluster.nodes:
            self._nodes.append(c_node)
        for _ in range(2):
            random.choice(self._nodes).activated = True
            random.choice(self._nodes).activated = False
        color_map = ['#33A6CC' if c_node.activated else 'gray'
                     for c_node in self._nodes]
        nx.draw(self._G, pos=self._nodes_pos, ax=self._nodes_ax,
                node_color=color_map)
        self._nodes_ax.figure.canvas.draw()

    def set_load(self, load):
        self._load = load

    def set_cluster(self, value):
        self._cluster = value

    def dynamic_load(self):
        while True:
            self.set_load(random.randint(1, 10))
            time.sleep(1)

    # def setNodePanelPic(self, pic: str):
    #     pixmap = QtGui.QPixmap(pic)
    #     pixmap = pixmap.scaled(self.loadPanel.width(), self.loadPanel.height())
    #     self.nodesPanel.setPixmap(pixmap)
    #
    # def setLoadPanelPic(self, pic: str):
    #     pixmap = QtGui.QPixmap(pic)
    #     pixmap = pixmap.scaled(self.loadPanel.width(), self.loadPanel.height())
    #     self.loadPanel.setPixmap(pixmap)
    @property
    def load_timer(self):
        return self._load_timer

    @property
    def nodes_timer(self):
        return self._nodes_timer


class MagnoneCtrl(object):
    """Magnone Controller class"""

    def __init__(self, view: MagnoneUi):
        self._simulating = False  # indicating whether the simulation is ON
        self.workerThreads = []
        self._view = view
        self._connectSignals()

    def startSim(self):
        self._view.load_timer.start()
        self._view.nodes_timer.start()
        # self._simulating = True
        # updateLoadPanelT = threading.Thread(target=self._updateLoadPanel)
        # updateNodesPanelT = threading.Thread(target=self._updateNodesPanel)
        # self.workerThreads.append(updateLoadPanelT)
        # self.workerThreads.append(updateNodesPanelT)
        # updateNodesPanelT.start()
        # updateLoadPanelT.start()

    def stopSim(self):
        self._view.load_timer.stop()
        self._view.nodes_timer.stop()
        """stop the two updating thread"""
        # self._simulating = False
        # for t in self.workerThreads:
        #     t.join()
        # print("Simulation Stopped")

    def resetSim(self):
        """Reset all the status into initial"""
        self._view.setLoadPanelPic('img/load0.png')
        self._view.setNodePanelPic('img/nodes0.png')

    def _updateLoadPanel(self, interval=0.5) -> None:
        """update the load status in the loadPanel

        Args:
            interval: the interval (s) of updating
        """
        t = 1
        while self._simulating:
            self._view.setLoadPanelPic(pic=f'img/load_t.png')
            time.sleep(interval)
            t += 1

    def _updateNodesPanel(self, interval=0.5):
        """update the nodes status in the nodesPanel

        Args:
            interval: the interval (s) of updating
        """
        t = 1
        while self._simulating:
            self._view.setNodePanelPic(pic=f'img/nodes_t.png')
            time.sleep(interval)
            t += 1

    def _connectSignals(self):
        """Connect signals and slots """
        self._view.buttons['start'].clicked.connect(self.startSim)
        self._view.buttons['stop'].clicked.connect(self.stopSim)
        self._view.buttons['reset'].clicked.connect(self.resetSim)






if __name__ == '__main__':
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
        for _ in range(1):
            n = Node(node_id=node_id, cores=v[0], mem=v[1], core_gen=v[2])
            cluster.add_node(n)
            node_id += 1
    win = MagnoneUi(cluster=cluster, comm_pipe=None)
    win.show()
    magnoneCtrl = MagnoneCtrl(view=win)

    T = threading.Thread(target=win.dynamic_load)
    T.start()
    app.exec()
