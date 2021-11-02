import multiprocessing
import typing
from enum import Enum
import time


class CpuGen(Enum):
    """four types of processing node"""
    A = 100
    B = 200
    C = 300
    D = 400


class Node(object):
    """The specification of computing node

    Attributes:
        _cores: An integer count of the cpu cores in this node.
        _mem: An integer indicating the memory size of this node.
        _core_gen: An integer indicating the generation of the cpu core.
        _activated: A boolean indicating whether this node is scheduled
    """

    def __init__(self, cores: int = 1, mem: int = 1000, core_gen: CpuGen = CpuGen.A, activated: bool = False):
        """None initializer"""
        self._cores = cores
        self._mem = mem
        self._free_cores = cores
        self._free_mem = mem
        self._core_gen = core_gen
        self._activated = activated

    @property
    def cores(self):
        return self.cores

    @property
    def mem(self):
        return self._mem

    @property
    def free_cores(self):
        return self._free_cores

    @property
    def free_mem(self):
        return self._free_mem

    @property
    def core_gen(self):
        return self._core_gen

    @property
    def activated(self):
        return self._activated

    @activated.setter
    def activated(self, value):
        self._activated = value

    def __repr__(self):
        return str(self._activated)

    @free_cores.setter
    def free_cores(self, value):
        self._free_cores = value

    @free_mem.setter
    def free_mem(self, value):
        self._free_mem = value


class ModelIns(object):
    """A deployed model instance

    Attributes:
        _p_node: the node where the model instance is deployed on
        _is_replica: indicating whether the instance is a replica
        _requeue: the local queue for this model instance,
            containing the queries from global controller
        _cores: the cores possessed by the model instance
        _mem: the size of memory possessed by the model instance
        _capability: the instance's capability to process one query
            processing time per query
        _t_last: the time stamp when it process one query last time
    """

    def __init__(self, p_node: Node, cores: int, mem: int, capability: float, is_replica: bool = False):
        """Model instance initialize"""
        self._p_node = p_node
        self._is_replica = is_replica
        self._requeue = multiprocessing.Queue()
        self._cores = cores
        self._mem = mem
        self._capability = capability
        self._t_last = time.time()

    @property
    def p_node(self):
        return self._p_node

    @property
    def is_replica(self):
        return self._is_replica

    @property
    def requeue(self):
        return self._requeue

    @property
    def cores(self):
        return self._cores

    @property
    def mem(self):
        return self.mem

    @property
    def capability(self):
        return self._capability

    @property
    def t_last(self):
        return self._t_last

    def process(self):  # TODO model instance process should be multiprocessing
        self._requeue.get()
        time.sleep(self._capability)
        self._t_last = time.time()

