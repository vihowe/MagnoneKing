import math
import random
import numpy as np
import collections
import time
from src.model import node
import networkx as nx
import matplotlib
from matplotlib import pyplot as plt


def simulate_load():
    shown_values = collections.deque(maxlen=100)
    my_dpi = 180
    fig = plt.figure(figsize=(1200 / my_dpi, 400 / my_dpi))
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(0, 100, 1)
    while True:
        plt.cla()
        new_value = random.randint(1, 101)
        shown_values.append(new_value)
        ax.plot(x[:len(shown_values)], list(shown_values))
        plt.savefig(f'img/load_t.png', bbox_inches='tight', dpi=my_dpi)
        time.sleep(1)


def simulate_cluster():
    nodes = []
    for i in range(20):
        nodes.append(node.Node())

    edges = []
    for i in range(55):
        node_i = random.choice(nodes)
        node_j = random.choice(nodes)
        if node_i != node_j:
            edges.append((node_i, node_j))

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    my_dpi = 180
    fig = plt.figure(figsize=(1200 / my_dpi, 600 / my_dpi))
    ax = fig.add_subplot(1, 1, 1)

    pos = nx.spring_layout(G, k=1)
    while True:
        plt.cla()
        for _ in range(2):
            random.choice(nodes).set_activated(True)
            random.choice(nodes).set_activated(False)
        color_map = ['#33A6CC' if c_node.get_state() else 'gray' for c_node in nodes]
        nx.draw(G, pos=pos, ax=ax, node_color=color_map)
        plt.pause(1)


if __name__ == '__main__':
    # simulate_load()
    simulate_cluster()
