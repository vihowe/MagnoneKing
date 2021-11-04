import multiprocessing
import queue
from enum import Enum
import time


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

    def __lt__(self, other):
        if isinstance(other, Request):
            return self._t_arri + self._t_slo < other._t_arri + other._t_slo
        else:
            return True     # signal STOP should be the last one in the priority queue

    # def __le__(self, other):
    #     if isinstance(other, Request):
    #         return self._t_arri + self._t_slo <= other._t_arri + other._t_slo
    #     else:
    #         return True

    def __repr__(self):
        return f'Request(id={self._r_id}, t_arri={self._t_arri}), ' \
               f't_slo={self._t_slo}, t_end={self.t_end}'

    def __str__(self):
        return f'({self._r_id}, {self._t_arri}, {self._t_slo}, ' \
               f'{self._t_end})'


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

    def __init__(self, cores: int = 1, mem: int = 1000, core_gen: CpuGen = CpuGen.A,
                 activated: bool = False):
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


class ModelIns(multiprocessing.Process):
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
        recv_pipe: the pipe for receiving request from the controller
        _t_last: the time stamp when it process one query last time
        _pri_queue: the priority queue for EDL scheduling
    """

    def __init__(self, p_node: Node, cores: int, mem: int, capability: float, recv_pipe: multiprocessing.Pipe,
                 is_replica: bool = False):
        """Model instance initialize"""
        super().__init__()
        self._p_node = p_node
        self._is_replica = is_replica
        self._requeue = multiprocessing.Queue()
        self._cores = cores
        self._mem = mem
        self._capability = capability
        self._t_last = time.time()
        self.recv_pipe = recv_pipe
        self._pri_queue = queue.PriorityQueue()

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

    def run(self):
        # TODO using EDF strategy to schedule reqs in local queue and do processing
        while True:
            q_size = self._requeue.qsize()
            # get all queries from queue and sort them by end line
            for _ in range(q_size):
                req = self._requeue.get()
                self._pri_queue.put(req)
            if not self._pri_queue.empty():
                req = self._pri_queue.get()
            else:
                continue

            if isinstance(req, Request):
                time.sleep(self._capability)
                self._t_last = time.time()
            elif req == -1:
                break
