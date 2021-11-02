import collections
import time
from enum import Enum

from typing import Tuple

from src.model import node


class Request(object):
    """magnetic detection request"""

    def __init__(self, r_id, t_arri: float, t_slo: float, t_end=-1):
        self._r_id = r_id
        self._t_arri = t_arri
        self._t_slo = t_slo
        self._t_end = t_end

    @property
    def r_id(self):
        return self._r_id

    @property
    def t_slo(self):
        return self._t_slo

    @t_slo.setter
    def t_slo(self, value):
        self._t_slo = value

    @property
    def t_arri(self):
        return self._t_arri

    @property
    def t_end(self):
        return self._t_end

    @t_end.setter
    def t_end(self, value):
        self._t_end = value


class Cluster(object):
    def __init__(self):
        self._nodes = []

    def add_node(self, c_node: node.Node):
        self._nodes.append(c_node)

    @property
    def nodes(self):
        return self._nodes


class Controller(object):
    """The controller for load balancing and auto scaling

    Attributes:
        _cluster: the cluster that the controller is in charge
        _g_queue: the queue for storing all user queries
        _model_insts: the current model instances in the cluster
        _slo: the service level objective
    """

    def __init__(self, cluster: Cluster, slo: float = 3000):
        self._cluster = cluster
        self._g_queue = collections.deque()
        self._model_insts = []
        self._slo = slo
        self._threshold = 0.9

    @property
    def slo(self):
        return self._slo

    def find_model_inst(self) -> node.ModelIns or None:
        """find the model instance which owns the lowest relative load
            return None if all the instances' relative load exceed the threshold
        """
        r_load = [len(model.requeue) * model.capabiity for model in self._model_insts]
        min_r = min(range(len(r_load)), key=r_load.__getitem__)
        if r_load[min_r] >= self._threshold * self._slo:
            return None
        else:
            return self._model_insts[min_r]

    def find_light_node(self) -> node.Node:
        """find an node which owns the greatest amount of resource"""

        for c_node in self._cluster.nodes:
            if not c_node.activated:
                return c_node

        scores = [c_node.free_cores * c_node.core_gen.value
                  + c_node.free_mem for c_node in self._cluster.nodes]
        return self._cluster.nodes[max(range(len(scores)), key=scores.__getitem__)]

    def find_resource(self, c_node: node.Node) -> Tuple[int, int, float]:
        """return the just enough resource when model is deployed on `c_node`
        and the model instance's capability. (processing time per query)
        """
        # TODO find the just enough resource allocated to this model instance
        return 1, 100, 1.

    def deploy_model_inst(self) -> node.ModelIns:
        """deploy a new model instance in an appropriate node and allocate
        just enough resource to it
        """
        c_node = self.find_light_node()
        cores, mem, cap = self.find_resource(c_node)
        c_node.free_cores -= cores
        c_node.free_mem -= mem
        c_node.activated = True

        model_inst = node.ModelIns(c_node, cores=cores, mem=mem, capability=cap)
        self._model_insts.append(model_inst)
        return model_inst

    def dispatch(self):
        """dispatch the user's query in the global queue to an appropriate
        model instance
        """
        req = self._g_queue.pop()
        a_model_inst = self.find_model_inst()
        if a_model_inst is None:    # all the model instances are under peak load
            a_model_inst = self.deploy_model_inst()
        a_model_inst.requeue.append(req)
        # TODO send the req to the model instance's query queue

    def monitoring(self, timeout, interval):  # TODO this should be multiprocessing
        """keep a lookout over all model instances, and clean model
        instance which is idle for a while"""
        model_inst: node.ModelIns
        for model_inst in self._model_insts:
            if time.time() - model_inst.t_last > timeout:
                # free resource
                p_node = model_inst.p_node
                p_node.free_cores += model_inst.cores
                p_node.free_mem += model_inst.mem
                self._model_insts.remove(model_inst)
        time.sleep(interval)
