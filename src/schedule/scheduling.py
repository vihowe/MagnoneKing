import bisect
import collections
import logging
import multiprocessing
import os
import pandas as pd
import queue
import random
import sys
import threading
import time
from typing import Dict, List, Tuple

sys.path.append('/home/vihowe/project/MagnoneKing/')
import src.schedule.profiling as profiling
from src.model.component import (Cluster, CpuGen, ModelIns, Node, Request,
                                 TaskType)

MODE = 0
DOCKER_COLD_START = 2.856
VM_COLD_START = 11.707


def report_lat(ret_queue: multiprocessing.Queue, avg_latency: multiprocessing.Value,
               instant_lat: multiprocessing.Queue, mode, interval=1):
    """report the avg latency of all finished queries
    Args:
        ret_queue: get all finished request from model instances
        avg_latency: the average latency shared with the controller
    """
    ret_dict = collections.defaultdict(list)
    avg_lat = 0.
    req_num = 0
    t_start = time.time()
    f = open(f'../latency_{mode}.log', 'w+')
    try:
        while True:
            req: Request
            req = ret_queue.get()
            if isinstance(req, Request):
                ret_dict[req.r_id].append(req.t_end - req.t_arri)
                if len(ret_dict[req.r_id]) == len(TaskType.__members__):
                    avg_lat = (avg_lat * req_num + max(ret_dict[req.r_id])) / (req_num + 1)
                    instant_lat.value = max(ret_dict[req.r_id]) 
                    f.write(str(instant_lat.value)+'\n')
                    avg_latency.value = avg_lat   # update the shared value
                    if time.time() - t_start >= interval:
                        logging.info(f'finished {req_num}requests, avg latency: {avg_lat}, instant latency: {instant_lat.value}')
                        t_start = time.time()
                    req_num += 1
                    ret_dict.pop(req.r_id)
        # else:     # TODO why if this process exit, the model instance cannot receive another request
        #     logging.debug('report avg latency process exit')
        #     break
    except Exception as e:
        print(e)
        f.close()


class Controller(multiprocessing.Process):
    """The controller for load balancing and auto scaling

    Attributes:
        _cluster: the cluster that the controller is in charge, a list of nodes
        _g_queue: the queue for storing all user queries
        _model_insts: a dict storing all kinds of model instances.
            _model_insts[task_type] = List[model_instance]
        _slo: the general service level objective. The query with high
            priority may have more stringent slo.
        recv_pipe: the pipe for communicating with user agent
        send_pipe_dic: store all the communication pipe with model instances
        ret_queue: store the results from all model instance
        launching: the number of model instance that is launching
        mode: using container or virtual machine(0-container, 1-vm)
        t1: the running time for long task when deployed on a strong core
        t2: the running time for long task when deployed on a weak core
    """

    def __init__(self, cluster: Cluster, recv_pipe: multiprocessing.Pipe, g_queue: multiprocessing.Queue,
                 slo: float = 1, mode=0):

        super().__init__()
        self._cluster = cluster
        self._g_queue = g_queue
        self._model_insts = collections.defaultdict(list)
        self._slo = slo
        self._threshold = 1
        self.recv_pipe = recv_pipe  # receiving msg from user agent
        self.send_pipe_dic = {}  # send request to appropriate model instance
        self.ret_queue = multiprocessing.Queue()
        self._task_affinity = None
        self.avg_lat = multiprocessing.Value('d', 0.0)
        self.instant_lat = multiprocessing.Value('d', 0.0)
        self._launching = 0
        self._launch_lock = threading.Lock()    # the lock for checking enough resource
        self.mode = mode
        self.cold_start = DOCKER_COLD_START if self.mode == 0 else VM_COLD_START
        self.t1 = 100
        self.t2 = 500

        self.profile_cluster()


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
        nodes = self._cluster.nodes
        ret = profiling.get_res_time(self._cluster)
        rret = {}
        for k, v in ret.items():    # key: task; value: list[tuple(cpu_share, mem_share, estimated_time)]
            r = []
            for item in v:
                r.append((nodes[item[0]-1], item[1:]))
            rret[k] = r
        self.t1 = min(rret[TaskType.B][0][-1][-1], rret[TaskType.D][0][-1][-1])
        self.t2 = min(rret[TaskType.B][-1][-1][-1], rret[TaskType.D][-1][-1][-1])
        return rret

    def profile_cluster(self):
        """Have a perception of the structure of this cluster
        return the probability `p` of assigning the long job to a strong core 

        So the controller will assign less resource to make the short job
        running about the same time as a long job on a strong core 
        with probability `p` and the same time as a long job on a weak core with probability `1-p`

        Return:
            p: the probability of assigning a long job to a strong core
        """
        total_core = 0
        strong_core = 0
        for node in self._cluster.nodes:
            if node.core_gen == CpuGen.A or node.core_gen == CpuGen.B:
                total_core += node.cores
                strong_core += node.cores
            else:
                total_core += node.cores
        
        p = strong_core * 4 / total_core
        self._strong_p = p if p < 1 else 1

    def find_model_inst(self, t_type: TaskType) -> ModelIns or None:
        """find the model instance which is responsible for `t_type` task and
          owns the highest relative load in the range of agreeing with SLO

          If all model instances are in a relative high load level, it needs 
          to launch a new model instance.

        Args:
            t_type: the type of task to process

        Return:
            (best_candidate_node, if_need_new_instance)
        """
        if len(self._model_insts[t_type]) == 0:
            return None
        r_load = [(pow((model.queue_size.value + model.requeue.qsize()), 2)
                   + (model.queue_size.value + model.requeue.qsize())) / 2
                  * model.t_cost for model in self._model_insts[t_type]]

        rr = sorted(r_load)
        slo_load = self._threshold * self._slo
        a = bisect.bisect_left(rr, slo_load)  # the qualified relative load candidate
        if a == 0:
            return self._model_insts[t_type][r_load.index(rr[0])], True
        return self._model_insts[t_type][r_load.index(rr[a - 1])], False

    def balance_task(self, cpu_gen: CpuGen, task_type: TaskType):
        """slice the resource on `cpu_gen` further to prolong the running time 
        of short task `task_type`

        Args:
            cpu_gen: the generation of cpu for handling this task
            task_type: task A or task C
            t1: the expected running time of this short task(similar to long job running on a strong core)
            t2: the expected running time of this short task(similar to long job running on a weak core)
            t2 must be greater than t1
        
        Return:
            (cpu_share, mem_share, estimated_time similar to long job running on strong core), (cpu_share, mem_share, estimated_time similar to long job running on weak core)
        """
        df = pd.read_csv(os.path.join('data', f'{cpu_gen.value}.csv'), header=None)
        df = df[df['task'] == task_type.value]
        p1 = None
        p2 = None
        for row in df.iterrows():
            running_time = row[1]['running time']
            if running_time < self.t2 * 1.2 and p2 is None:
                p2 = (row[1]['cpu_quota'], 100, running_time)
                continue
            if running_time < self.t1 * 1.2 and p1 is None:
                p1 = (row[1]['cpu_quota'], 100, running_time)
                break
        return p1, p2
            


    def find_light_node(self, task_type: TaskType, mode) -> Node:
        """find an node which is affinity to `task_type` model instance;
        slice its cpu again to prolong its running time if its taskA or taskC

        Args:
            task_type: the type of model instance
            mode: if slice the cpu further(0-do not slice, 1-slice)
        Return:
            (Node, estimated_rs_time): an appropriate node to bear this model instance
                and its resource allocation and estimated processing time for running this task
            if there are no free resource, return None
        """
        for c_node, item in self._task_affinity[task_type]:

            if mode == 1:
                cpu_gen = c_node.core_gen

                if task_type == TaskType.A or task_type == TaskType.C:
                    p1, p2 = self.balance_task(cpu_gen, task_type)
                    assert p1 is not None
                    assert p2 is not None
                
                item = random.choices([p1, p2], weights=[self._strong_p, 1-self._strong_p])

            if c_node.free_cores >= item[0] and c_node.free_mem >= item[1]:
                return (c_node, item)
        return None, None


    def deploy_model_inst(self, task_type: TaskType) -> ModelIns:
        """deploy a new model instance which processing `task_type` task
        in an appropriate node and allocate just enough resource to it

        Args:
            task_type: the type of this model instance
        
        Return:
            return a new model instance, if no free resource to build one, return None.
        """

        # find an appropriate node and resource amount
        self._launch_lock.acquire()
        c_node, resource_allo = self.find_light_node(task_type)
        if c_node is None:
            return None
        # print(c_node, resource_allo)
        # time.sleep(3)
        # cores, mem, cap = self.find_resource(c_node, task_type)
        cores = resource_allo[0]
        mem = resource_allo[1]
        cap = resource_allo[2]

        c_node.free_cores -= cores
        c_node.free_mem -= mem
        self._launch_lock.release()

        time.sleep(self.cold_start)

        self._launch_lock.acquire()
        c_node.container_num += 1
        c_node.activated = True
        self._launch_lock.release()

        # start and register this model instance
        parent, child = multiprocessing.Pipe()
        model_inst = ModelIns(t_type=task_type, p_node=c_node, cores=cores, mem=mem, t_cost=cap, recv_pipe=child,
                              ret_queue=self.ret_queue)
        
        model_inst.start()

        self.send_pipe_dic[model_inst] = parent
        self._model_insts[task_type].append(model_inst)
        self._launching -= 1

        return model_inst

    def dispatch(self, req: Request):
        """dispatch the user's query in the global queue to an appropriate
        model instance of each TaskType, do autoscaling simultaneously
        """

        # dispatch this request to all four types of tasks simultaneously
        for _, t_type in TaskType.__members__.items():
            a_model_inst, new_flag = self.find_model_inst(t_type)
            if new_flag is True and self._launching < 2:  # need to launch a new docker container
                # using another thread to start the container
                print(f'***************deploy new model instance')
                self._launching += 1
                deploy_new_t = threading.Thread(target=self.deploy_model_inst, args=(t_type,), daemon=True)
                deploy_new_t.start()
                # new_model_ins = self.deploy_model_inst(t_type)
                # if new_model_ins is not None:
                #     a_model_inst = new_model_ins
            a_model_inst.requeue.put(req)

    def monitoring(self, timeout=20, interval=5):
        """keep looking all model instances, and clean model
        instance which has been idle for a while

        Args:
            timeout: to kill the model instance if its idle time is too long (s).
            interval: the interval (s).
        """
        while True:
            for k in list(self._model_insts):
                for model_inst in self._model_insts[k]:
                    if time.time() - model_inst.t_last > timeout:
                        # free resource
                        p_node = model_inst.p_node
                        p_node.free_cores += model_inst.cores
                        p_node.free_mem += model_inst.mem
                        p_node.container_num -= 1
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
            keys = list(self._model_insts)
            for k in keys:
                model_num += len(self._model_insts[k])
            logging.info(f'There are {model_num} model instance serving')
            for k in keys:
                for item in self._model_insts[k]:
                    logging.info(
                        f'\t{item.name}: type:{item.t_type}, p_node: {item.p_node.node_id}, avg_latency: {item.avg_latency.value}, served {item.req_num.value} requests')
            time.sleep(interval)

    def report_cluster(self):
        """Report the cluster status, avg latency and instant latency to User Agent"""
        while True:
            msg = self.recv_pipe.recv()
            if msg == 'cluster':
                self.recv_pipe.send((self._cluster, self.avg_lat.value, self.instant_lat.value))


    def run(self) -> None:
        # Once the controller starts, it profiles all type of tasks to get the affinity data
        # and deploys one set of model instance in the cluster
        self._task_affinity = self.profile()
        for _, t_type in TaskType.__members__.items():
            t = threading.Thread(target=self.deploy_model_inst, args=(t_type, ))
            t.start()
        t.join()

        # start the monitor thread
        monitor = threading.Thread(target=self.monitoring, args=(25, 10), daemon=True)
        monitor.start()

        # start the report thread
        reporter = threading.Thread(target=self.report, args=(5,), daemon=True)
        reporter.start()

        T_report_cluster = threading.Thread(target=self.report_cluster, daemon=True)
        T_report_cluster.start()

        # start report avg latency process
        p = multiprocessing.Process(target=report_lat, args=(self.ret_queue, self.avg_lat, self.instant_lat, self.mode), daemon=True)
        p.start()

        # start dispatching user queries
        while True:
            req = self._g_queue.get()
            if isinstance(req, Request):
                self.dispatch(req)
            else:
                self.ret_queue.put(-1)
                logging.debug('receieved the stop signal')
                sys.exit()


class UserAgent(multiprocessing.Process):
    """User agent is responsible for sending queries to the controller

    Args:
        cluster: the cluster that processing queries
        config: some running specification
        load: the current load
        mode: using vm or container(0-container, 1-vm)
    """

    def __init__(self, cluster: Cluster, comm_pipe: multiprocessing.Pipe,
                 config: Dict = None, load=0.1, mode=0):
        super().__init__()
        self.comm_pipe = comm_pipe  # the pipe for communication with the view
        self.send_pipe = None   # the pipe for communication with the controller
        self.cluster = cluster
        self._run_config = config
        self.load = load
        self.latest_rid = 0
        self.mode = mode 

    def start_up(self):
        parent, child = multiprocessing.Pipe()
        self.send_pipe = parent
        self.g_queue = multiprocessing.Queue()
        controller = Controller(cluster=self.cluster, recv_pipe=child, g_queue=self.g_queue, slo=self._run_config['slo'], mode=self.mode)
        controller.start()

    def querying(self, total_queries=5000):
        """sending query request to the controller

        Args:
            load: the simulated query load (qps)
            total_queries: the number of all queries to be sent
        """
        r_id = 0
        # load_file = open('load.config','r')
        # self.load = int(load_file.readline()) * 3
        self.load = 10
        t_tick = time.time()
        load = [40, 120, 80, 0]
        load_i = 0
        while True:
            if time.time() - t_tick > 30:
                t_tick = time.time()
                # load = load_file.readline()
                # if load != '':
                #     self.load = int(load) * 3
                # else:
                #     self.load = 0
                #     break
                self.load = load[load_i]
                load_i += 1 if load_i < 3 else 0 
            req = Request(r_id, time.perf_counter(), 1)
            self.g_queue.put(req)
            self.latest_rid = req.r_id
            r_id += 1

            time.sleep(random.expovariate(self.load))

    def report(self, interval=1):
        """report the current cluster status, load, avg_latency and instant_latency to view"""
        start = 0
        while True:
            self.send_pipe.send('cluster')
            clu, avg_latency, instant_latency = self.send_pipe.recv()
            # print(clu.nodes, self.load)
            latest_rid = self.latest_rid
            self.comm_pipe.send((clu, latest_rid-start, avg_latency, instant_latency))
            start = latest_rid
            time.sleep(interval)

    def run(self) -> None:
        self.start_up()
        time.sleep(20)
        T = threading.Thread(target=self.report, args=(1, ))
        T.start()
        self.querying()
        logging.debug('querying finished')


def main():
    # cluster initialization
    cluster = Cluster()

    node_specification = {
        "desktop": (8, 8192, CpuGen.A,),
        "laptop": (4, 4096, CpuGen.B,),
        "pi": (2, 2048, CpuGen.C,),
    }
    node_id = 1
    # for v in node_specification.values():
    #     for _ in range(2):
    #         n = Node(node_id=node_id, cores=v[0], mem=v[1], core_gen=v[2])
    #         cluster.add_node(n)
    #         node_id += 1
    s1 = node_specification['laptop']
    s2 = node_specification['pi']
    for _ in range(2):
        n = Node(node_id=node_id, cores=s1[0], mem=s1[1], core_gen=s1[2])
        cluster.add_node(n)
        node_id += 1
    for _ in range(6):
        n = Node(node_id=node_id, cores=s2[0], mem=s2[1], core_gen=s2[2])
        cluster.add_node(n)
        node_id += 1

    config = {
        'slo': 1,
    }
    user_agent = UserAgent(cluster, None, config)
    # user_agent.start_up()
    # user_agent.querying()
    user_agent.start()
    user_agent.join()
    logging.debug('user agent exit')
    # user_agent.run()


def test():
    pass


if __name__ == '__main__':
    # test()
    root = logging.getLogger()
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(processName)-8s %(name)s %(levelname)-6s %(message)s')
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)
    root.setLevel(logging.DEBUG)


    main()
