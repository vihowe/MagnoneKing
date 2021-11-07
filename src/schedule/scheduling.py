import collections
import queue
import threading
import time
import multiprocessing
import random
import bisect

from typing import Tuple, Dict

import sys
sys.path.append('/home/vihowe/project/MagnoneKing/')

from src.model.component import CpuGen, Node, ModelIns, Request


STOP_SLO = 1 << 30


class Cluster(object):
    def __init__(self):
        self._nodes = []

    def add_node(self, c_node: Node):
        self._nodes.append(c_node)

    @property
    def nodes(self):
        return self._nodes


class Controller(multiprocessing.Process):
    """The controller for load balancing and auto scaling

    Attributes:
        _cluster: the cluster that the controller is in charge
        _g_queue: the queue for storing all user queries
        _model_insts: the current model instances in the cluster
        _slo: the general service level objective. The query with high
            priority may have more stringent slo.
        recv_pipe: the pipe for communicating with user agent
    """

    def __init__(self, cluster: Cluster, recv_pipe: multiprocessing.Pipe, g_queue: multiprocessing.Queue, slo: float = 3):

        super().__init__()
        self._cluster = cluster
        self._g_queue = g_queue
        self._model_insts = []
        self._slo = slo
        self._threshold = 0.7
        self.recv_pipe = recv_pipe  # receiving msg from user agent
        self.send_pipe_dic = {}  # send request to appropriate model instance

    @property
    def slo(self):
        return self._slo

    def find_model_inst(self) -> ModelIns or None:
        """find the model instance which owns the highest relative load in the range of agreeing with SLO
            return None if all the instances' relative load exceed the threshold
        """
        if len(self._model_insts) == 0:
            return None
        r_load = [(model.queue_size.value + model.requeue.qsize()) * model.capability for model in self._model_insts]
        # print(r_load)

        slo_load = self._threshold * self._slo
        rr = sorted(r_load)
        a = bisect.bisect_left(rr, slo_load)    # the qualified relative load candidate
        if a == 0:
            return None
        return self._model_insts[r_load.index(rr[a-1])]

        # min_r = min(range(len(r_load)), key=r_load.__getitem__)
        # if r_load[min_r] >= self._threshold * self._slo:
        #     return None
        # else:
        #     return self._model_insts[min_r]

    def find_light_node(self) -> Node:
        """find an node which owns the greatest amount of resource"""

        for c_node in self._cluster.nodes:
            if not c_node.activated:
                return c_node

        scores = [c_node.free_cores * c_node.core_gen.value
                  + c_node.free_mem for c_node in self._cluster.nodes]
        return self._cluster.nodes[max(range(len(scores)), key=scores.__getitem__)]

    def find_resource(self, c_node: Node) -> Tuple[int, int, float]:
        """return the just enough resource when model is deployed on `c_node`
        and the model instance's capability. (processing time per query)
        """
        # TODO find the just enough resource allocated to this model instance
        return 1, 100, 1.

    def deploy_model_inst(self) -> ModelIns:
        """deploy a new model instance in an appropriate node and allocate
        just enough resource to it
        """
        c_node = self.find_light_node()
        cores, mem, cap = self.find_resource(c_node)
        c_node.free_cores -= cores
        c_node.free_mem -= mem
        c_node.activated = True

        parent, child = multiprocessing.Pipe()
        model_inst = ModelIns(c_node, cores=cores, mem=mem, capability=cap, recv_pipe=child)
        model_inst.start()
        self.send_pipe_dic[model_inst] = parent
        self._model_insts.append(model_inst)

        return model_inst

    def dispatch(self, req: Request):
        """dispatch the user's query in the global queue to an appropriate
        model instance, do autoscaling simultaneously
        """
        a_model_inst = self.find_model_inst()
        if a_model_inst is None:  # all the model instances are under peak load
            a_model_inst = self.deploy_model_inst()
        # print(f'****request {req} is sent to {a_model_inst}')
        a_model_inst.requeue.put(req)

    def monitoring(self, timeout=20, interval=20):
        """keep a lookout over all model instances, and clean model
        instance which is idle for a while

        Args:
            timeout: to kill the model instance if its idle time is too long (s).
            interval: the interval (s).
        """
        while True:
            for model_inst in self._model_insts:
                if time.time() - model_inst.t_last > timeout:
                    # free resource
                    p_node = model_inst.p_node
                    p_node.free_cores += model_inst.cores
                    p_node.free_mem += model_inst.mem
                    # send signal to model instance for exiting
                    model_inst.requeue.put(Request(-1))
                    self._model_insts.remove(model_inst)
            time.sleep(interval)

    def report(self, interval=5):
        """report the status of model instances and the average latency

        Args:
            interval: the interval between reporting
        """
        while True:
            a = list(self._model_insts)
            print(f'==There are {len(a)} model instance ==')
            for item in a:
                print(f'\t{item}: p_node: {item.p_node.node_id}, avg_latency: {item.avg_latency.value}')
            if len(a) == 0:
                print(f"No model instance is activated now")
            time.sleep(interval)

    def run(self) -> None:
        # Once the controller starts, it deploys one model instance in the cluster
        self.deploy_model_inst()
        # start the monitor thread
        monitor = threading.Thread(target=self.monitoring, args=(60, 30))
        monitor.start()

        # start the report thread
        reporter = threading.Thread(target=self.report, args=(10,))
        reporter.start()

        # start dispatching user queries
        while True:
            req = self._g_queue.get()
            if isinstance(req, Request):
                self.dispatch(req)
            else:
                break


class UserAgent(object):
    """User agent is responsible for sending queries to the controller

    Args:
        cluster: the cluster that processing queries
        _config: some running specification
    """

    def __init__(self, cluster: Cluster, config: Dict = None):
        self.send_pipe = None
        self.cluster = cluster
        self._config = config

    def start_up(self):
        parent, child = multiprocessing.Pipe()
        self.send_pipe = parent
        self.g_queue = multiprocessing.Queue()
        controller = Controller(cluster=self.cluster, recv_pipe=child, g_queue=self.g_queue, slo=self._config['slo'])
        controller.start()

    def querying(self, load, total_queries=10000):
        """sending query request to the controller

        Args:
            load: the simulated query load (qps)
            total_queries: the number of all queries to be sent
        """
        r_id = 0
        while True:
            if r_id == total_queries:
                break
            req = Request(r_id, time.perf_counter(), 3)
            r_id += 1
            self.g_queue.put(req)
            # time.sleep(1000)
            # print(f"request {r_id} have been sent.")
            if r_id < 1000:
                time.sleep(random.expovariate(3))
            elif r_id < 1500:
                time.sleep(random.expovariate(5))
            else:
                time.sleep(random.expovariate(100))


def main():
    # cluster initialization
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

    config = {
        'slo': 3,
    }
    user_agent = UserAgent(cluster, config)
    user_agent.start_up()
    user_agent.querying(load=5)


def test():
    pqueue = queue.PriorityQueue()
    for i in range(10):
        req = Request(i, time.time(), random.randint(1, 10))
        time.sleep(random.randint(1, 5))
        pqueue.put(req)
    pqueue.put(-1)

    for _ in range(pqueue.qsize()):
        print(pqueue.get())


if __name__ == '__main__':
    # test()
    main()