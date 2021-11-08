import multiprocessing
import queue
from enum import Enum
import time
from typing import MutableMapping


class Request(object):
    """magnetic detection request"""

    def __init__(self, r_id, t_arri: float = 0, t_slo: float = 0, t_end=-1):
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
            return True  # signal STOP should be the last one in the priority queue

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


class TaskType(Enum):
    """four types of tasks"""
    A = 0
    B = 1
    C = 2
    D = 3


class Node(object):
    """The specification of computing node

    Attributes:
        _cores: An integer count of the cpu cores in this node.
        _mem: An integer indicating the memory size of this node.
        _core_gen: An integer indicating the generation of the cpu core.
        _activated: A boolean indicating whether this node is scheduled
    """

    def __init__(self, node_id=1, cores: int = 1, mem: int = 1000, core_gen: CpuGen = CpuGen.A,
                 activated: bool = False):
        """None initializer"""
        self.node_id = node_id
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
        return f"Node('{self.node_id}', {self.free_cores}, {self.free_mem}, {self._core_gen}, {self.activated})"

    @free_cores.setter
    def free_cores(self, value):
        self._free_cores = value

    @free_mem.setter
    def free_mem(self, value):
        self._free_mem = value


class ModelIns(multiprocessing.Process):
    """A deployed model instance

    Attributes:
        _type: the type of the task it is running
        _p_node: the node where the model instance is deployed on
        _is_replica: indicating whether the instance is a replica
        _requeue: the local queue for this model instance,
            containing the queries from global controller
        _cores: the cores possessed by the model instance
        _mem: the size of memory possessed by the model instance
        _t_cost: the instance's capability to process one query
            processing time per query
        recv_pipe: the pipe for receiving msg from the controller
        _t_last: the time stamp when it process one query last time
        _pri_queue: the priority queue for EDL scheduling
        req_num: the number of served queries
        avg_latency: the average latency
    """

    def __init__(self, t_type, p_node: Node, cores: int, mem: int, t_cost: float, recv_pipe: multiprocessing.Pipe,
                 is_replica: bool = False, ):
        """Model instance initialize"""
        super().__init__()
        self._type = t_type
        self._p_node = p_node
        self._is_replica = is_replica
        self._requeue = multiprocessing.Queue()
        self._cores = cores
        self._mem = mem
        self._t_cost = t_cost
        self._t_last = multiprocessing.Value('d', time.time())
        self.recv_pipe = recv_pipe
        self._pri_queue = queue.PriorityQueue()
        self.req_num = multiprocessing.Value('i', 0)
        self.avg_latency = multiprocessing.Value('d', 0.0)
        self.queue_size = multiprocessing.Value('i', 0)

    # def __repr__(self) -> str:
    #     return f"{self._p_node}"

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
    def pri_queue(self):
        return self._pri_queue

    @property
    def cores(self):
        return self._cores

    @property
    def mem(self):
        return self._mem

    @property
    def t_cost(self):
        return self._t_cost

    @property
    def t_last(self):
        return self._t_last.value

    def run(self):
        # TODO using EDF strategy to schedule reqs in local queue and do processing
        while True:
            s = self._requeue.qsize()
            self.queue_size.value = self._pri_queue.qsize()

            # get all queries from queue and sort them by end line
            for _ in range(s):
                req = self._requeue.get()
                self._pri_queue.put(req)

            if not self._pri_queue.empty():
                req = self._pri_queue.get()
            else:
                continue

            req: Request
            if req.r_id != -1:
                time.sleep(self._t_cost)
                req.t_end = time.perf_counter()

                self.avg_latency: multiprocessing.Value
                with self.avg_latency.get_lock():
                    # print(self.avg_latency)
                    self.avg_latency.value = (self.avg_latency.value * self.req_num.value +
                                              req.t_end - req.t_arri) / (self.req_num.value + 1)
                with self.req_num.get_lock():
                    self.req_num.value += 1
                with self._t_last.get_lock():
                    self._t_last.value = time.time()
            else:
                print(f"Model Instance {self} has exited")
                break
