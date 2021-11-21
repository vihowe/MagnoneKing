import collections
import queue
import threading
import time
import multiprocessing
import random
import bisect

from typing import Tuple, Dict, List

import sys

import src.schedule.profiling as profiling

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


def report_lat(ret_queue: multiprocessing.Queue, interval=5):
    """report the avg latency of all finished queries"""
    ret_dict = collections.defaultdict(list)
    avg_lat = 0.
    req_num = 0
    t_start = time.time()
    while True:
        req: Request
        req = ret_queue.get()
        ret_dict[req.r_id].append(req.t_end - req.t_arri)
        if len(ret_dict[req.r_id]) == len(TaskType.__members__):
            avg_lat = (avg_lat * req_num + max(ret_dict[req.r_id])) / (req_num + 1)
            if time.time() - t_start >= interval:
                print(f'***************avg latency: {avg_lat}*******************')
                t_start = time.time()
            req_num += 1
            ret_dict.pop(req.r_id)


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

    def __init__(self, cluster: Cluster, recv_pipe: multiprocessing.Pipe, g_queue: multiprocessing.Queue,
                 slo: float = 1):

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
        # ret = {}
        nodes = self._cluster.nodes
        nodes_id = []
        for c_node in nodes:
            nodes_id.append(c_node.node_id)
        # for _, t_type in TaskType.__members__.items():
        #     ordered_nodes = random.sample(nodes, k=len(nodes))
        #     resource_cap = [(1, random.randint(20, 100), random.uniform(0.05, 0.1)) for _ in range(len(nodes))]
        #     ret[t_type] = list(zip(ordered_nodes, resource_cap))
        # return ret
        ret = profiling.get_res_time(nodes_id)
        rret = {}
        for k, v in ret.items():    # key: task; value: list[tuple]
            r = []
            for item in v:
                r.append((nodes[item[0]], *item[1:]))
            rret[k] = r
        return rret


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
        a = bisect.bisect_left(rr, slo_load)  # the qualified relative load candidate
        if a == 0:
            return None
        return self._model_insts[t_type][r_load.index(rr[a - 1])]

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
        # print(self._task_affinity[task_type])
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
        # print(c_node, resource_allo)
        # time.sleep(3)
        # cores, mem, cap = self.find_resource(c_node, task_type)
        cores = resource_allo[0]
        mem = resource_allo[1]
        cap = resource_allo[2]

        c_node.free_cores -= cores
        c_node.free_mem -= mem
        c_node.activated = True

        parent, child = multiprocessing.Pipe()
        model_inst = ModelIns(t_type=task_type, p_node=c_node, cores=cores, mem=mem, t_cost=cap, recv_pipe=child,
                              ret_queue=self.ret_queue)
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

    def monitoring(self, timeout=5, interval=5):
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
                        if p_node.free_cores == p_node.cores:
                            p_node.activated = False
                        # send signal to model instance for exiting
                        model_inst.requeue.put(Request(-1))
                        self._model_insts[k].remove(model_inst)
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
            for k, v in self._model_insts.items():
                for item in v:
                    pass
                    # print(
                        # f'\t{item}: type:{item.t_type}, p_node: {item.p_node.node_id}, avg_latency: {item.avg_latency.value}, served {item.req_num.value} requests')
            time.sleep(interval)

    def report_cluster(self):
        while True:
            msg = self.recv_pipe.recv()
            if msg == 'cluster':
                self.recv_pipe.send(self._cluster)


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

        T_report_cluster = threading.Thread(target=self.report_cluster)
        T_report_cluster.start()

        # start report avg latency process
        p = multiprocessing.Process(target=report_lat, args=(self.ret_queue,), daemon=True)
        p.start()

        # start dispatching user queries
        while True:
            req = self._g_queue.get()
            if isinstance(req, Request):
                self.dispatch(req)
            else:
                break


class UserAgent(multiprocessing.Process):
    """User agent is responsible for sending queries to the controller

    Args:
        cluster: the cluster that processing queries
        config: some running specification
        load: the current load
    """

    def __init__(self, cluster: Cluster, comm_pipe: multiprocessing.Pipe,
                 config: Dict = None, load=0.1):
        super().__init__()
        self.comm_pipe = comm_pipe  # the pipe for communication with the view
        self.send_pipe = None
        self.cluster = cluster
        self._run_config = config
        self.load = load

    def start_up(self):
        parent, child = multiprocessing.Pipe()
        self.send_pipe = parent
        self.g_queue = multiprocessing.Queue()
        controller = Controller(cluster=self.cluster, recv_pipe=child, g_queue=self.g_queue, slo=self._run_config['slo'])
        controller.start()

    def querying(self, total_queries=1600):
        """sending query request to the controller

        Args:
            load: the simulated query load (qps)
            total_queries: the number of all queries to be sent
        """
        r_id = 0
        while True:
            if r_id == total_queries:
                self.load = 0
                break
            req = Request(r_id, time.perf_counter(), 3)
            r_id += 1
            self.g_queue.put(req)
            if r_id < 100:
                self.load = random.randint(5, 10)
            elif r_id < 500:
                self.load = random.randint(40, 60)
            elif r_id < 1500:
                self.load = random.randint(200, 220)
            else:
                self.load = random.randint(1, 10)
            time.sleep(random.expovariate(self.load))

    def report(self, interval=1):
        """report the current cluster status and load to view"""
        while True:
            self.send_pipe.send('cluster')
            clu = self.send_pipe.recv()
            # print(clu.nodes, self.load)
            self.comm_pipe.send((clu, self.load))
            time.sleep(interval)

    def run(self) -> None:
        self.start_up()
        T = threading.Thread(target=self.report, args=(1, ))
        T.start()
        self.querying()


def main():
    # cluster initialization
    cluster = Cluster()

    node_specification = {
        "desktop": (12, 8192, CpuGen.D,),
        "pi": (4, 2048, CpuGen.A,),
        "laptop": (10, 4096, CpuGen.C,),
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
    user_agent = UserAgent(cluster, None, config)
    # user_agent.start_up()
    # user_agent.querying()
    user_agent.start()
    user_agent.join()
    # user_agent.run()


def test():
    pass


if __name__ == '__main__':
    # test()
    main()
