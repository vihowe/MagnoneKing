import collections
import queue
import threading
import time
import multiprocessing
import random
import bisect

from typing import Tuple, Dict, List

import sys
sys.path.append('/home/vihowe/project/MagnoneKing/')

from src.model.component import CpuGen, Node, ModelIns, Request, TaskType


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
        _model_insts: a dict storing all kinds of model instances.
            _model_insts[task_type] = List[model_instance]
        _slo: the general service level objective. The query with high
            priority may have more stringent slo.
        recv_pipe: the pipe for communicating with user agent
        send_pipe_dic: store all the communication pipe with model instances
        ret_queue: store the results from all model instance
    """

    def __init__(self, cluster: Cluster, recv_pipe: multiprocessing.Pipe, g_queue: multiprocessing.Queue, slo: float = 3):

        super().__init__()
        self._cluster = cluster
        self._g_queue = g_queue
        self._model_insts = collections.defaultdict(list)
        self._slo = slo
        self._threshold = 0.7
        self.recv_pipe = recv_pipe  # receiving msg from user agent
        self.send_pipe_dic = {}  # send request to appropriate model instance
        self.ret_queue = multiprocessing.Queue()
        self._task_affinity = None

    @property
    def slo(self):
        return self._slo

    def profile(self) -> Dict[TaskType, List[Tuple[Node, Tuple[int, int, float]]]]:
        """Simulate the profiling process to find the affinity of each task to each node

        Return:
            For each type of task, return a list of nodes ordered by affinity priority cupled
            with the appropriate amount of resource (#cpus, mem)
            when the task is deployed on it, and the estimated processing time.
        """
        # TODO read from profiled file to generate this data structure
        ret = {}
        nodes = self._cluster.nodes
        for _, t_type in TaskType.__members__.items():
            ordered_nodes = random.sample(nodes, k=len(nodes))
            resource_cap = [(1, random.randint(20, 100), random.uniform(0.05, 0.1)) for _ in range(len(nodes))]
            ret[t_type] = list(zip(ordered_nodes, resource_cap))
        return ret
            
    def find_model_inst(self, t_type: TaskType) -> ModelIns or None:
        """find the model instance which is responsible for `t_type` task and
          owns the highest relative load in the range of agreeing with SLO

        Args:
            t_type: the type of task to process

        Return:
            return None if all the instances' relative load exceed the threshold
        """
        if len(self._model_insts[t_type]) == 0:
            return None
        r_load = [(pow((model.queue_size.value + model.requeue.qsize()), 2)
                   + (model.queue_size.value + model.requeue.qsize())) / 2
                  * model.t_cost for model in self._model_insts[t_type]]
        # print(r_load)

        slo_load = self._threshold * self._slo
        rr = sorted(r_load)
        a = bisect.bisect_left(rr, slo_load)    # the qualified relative load candidate
        if a == 0:
            return None
        return self._model_insts[t_type][r_load.index(rr[a-1])]

        # min_r = min(range(len(r_load)), key=r_load.__getitem__)
        # if r_load[min_r] >= self._threshold * self._slo:
        #     return None
        # else:
        #     return self._model_insts[min_r]

    def find_light_node(self, task_type: TaskType) -> Node:
        """find an node which is affinity to `task_type` model instance and
        owns the greatest amount of resource

        Args:
            task_type: the type of model instance
        Return:
            (Node, estimated_time): an appropriate node to bear this model instance
                and its resource allocation and estimated processing time for running this task
        """
        print(self._task_affinity[task_type])
        for c_node, item in self._task_affinity[task_type]:

            if c_node.free_cores >= item[0] and c_node.free_mem >= item[1]:
                return (c_node, item)
        # TODO what if all resource are run out
        # for c_node in self._cluster.nodes:
        #     if not c_node.activated:
        #         return c_node

        # scores = [0 if c_node.free_cores <= 0 or c_node.free_mem <= 0 else (c_node.free_cores * c_node.core_gen.value
        #           + c_node.free_mem) for c_node in self._cluster.nodes]
        # print(scores)
        # return self._cluster.nodes[max(range(len(scores)), key=scores.__getitem__)]

    def find_resource(self, c_node: Node, task_type: TaskType) -> Tuple[int, int, float]:
        """return the just enough resource when model is deployed on `c_node`
        and the model instance's capability. (processing time per query)

        Args:
            c_node: the node which have the model instance
            task_type: the type of this model instance
        """
        # TODO find the just enough resource allocated to this model instance
        return 1, 100, 1.

    def deploy_model_inst(self, task_type: TaskType) -> ModelIns:
        """deploy a new model instance which processing `task_type` task
        in an appropriate node and allocate just enough resource to it

        Args:
            task_type: the type of this model instance
        """

        # find an appropriate node and resource amount
        c_node, resource_allo = self.find_light_node(task_type)
        print(c_node, resource_allo)
        time.sleep(3)
        # cores, mem, cap = self.find_resource(c_node, task_type)
        cores = resource_allo[0]
        mem = resource_allo[1]
        cap = resource_allo[2]

        c_node.free_cores -= cores
        c_node.free_mem -= mem
        c_node.activated = True

        parent, child = multiprocessing.Pipe()
        model_inst = ModelIns(t_type=task_type, p_node=c_node, cores=cores, mem=mem, t_cost=cap, recv_pipe=child, ret_queue=self.ret_queue)
        model_inst.start()

        self.send_pipe_dic[model_inst] = parent
        self._model_insts[task_type].append(model_inst)

        return model_inst

    def dispatch(self, req: Request):
        """dispatch the user's query in the global queue to an appropriate
        model instance of each TaskType, do autoscaling simultaneously
        """

        # dispatch this request to all four types of tasks simultaneously
        for _, t_type in TaskType.__members__.items():
            a_model_inst = self.find_model_inst(t_type)
            if a_model_inst is None:  # all the model instances are under peak load
                a_model_inst = self.deploy_model_inst(t_type)
            # print(f'****request {req} is sent to {a_model_inst}')
            a_model_inst.requeue.put(req)

    def monitoring(self, timeout=10, interval=10):
        """keep a lookout over all model instances, and clean model
        instance which is idle for a while

        Args:
            timeout: to kill the model instance if its idle time is too long (s).
            interval: the interval (s).
        """
        while True:
            for k, v in self._model_insts.items():
                for model_inst in v:
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
            # a = list(self._model_insts)
            model_num = 0
            for k, v in self._model_insts.items():
                model_num += len(v)
            print(f'==There are {model_num} model instance ==')
            # for item in a:
            #     print(f'\t{item}: p_node: {item.p_node.node_id}, avg_latency: {item.avg_latency.value}, served {item.req_num.value} requests')
            # if len(a) == 0:
            #     print(f"No model instance is activated now")
            time.sleep(interval)

    def run(self) -> None:
        # Once the controller starts, it profiles all type of tasks to get the affinity data
        # and deploys one set of model instance in the cluster
        self._task_affinity = self.profile()
        for _, t_type in TaskType.__members__.items():
            self.deploy_model_inst(t_type)
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
            if r_id < 40:
                time.sleep(random.expovariate(1))
            elif r_id < 200:
                time.sleep(random.expovariate(20))
            else:
                time.sleep(random.expovariate(1))


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
        for _ in range(1):
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