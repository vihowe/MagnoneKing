import multiprocessing
import os
import time


class MainProcess:
    def __init__(self, main_process_time, child_process_time):
        self.main_process_time = main_process_time
        self.child_process_time = child_process_time

    def excutor(self):
        print('main process begin, pid={0}, ppid={1}'.format(os.getpid(), os.getppid()))
        p = ChildProcess(self.child_process_time)
        p.start()
        # p.join()
        for i in range(self.main_process_time):
            print('main process, pid={0}, ppid={1}, times={2}'.format(os.getpid(), os.getppid(), i))
            time.sleep(1)
        print('main process end, pid={0}, ppid={1}'.format(os.getpid(), os.getppid()))


class ChildProcess(multiprocessing.Process):
    def __init__(self, process_time):
        multiprocessing.Process.__init__(self)
        self.process_time = process_time

    def run(self):
        print('child process begin, pid={0}, ppid={1}'.format(os.getpid(), os.getppid()))
        for i in range(self.process_time):
            print('child process pid={0}, ppid={1}, times={2}'.format(os.getpid(), os.getppid(), i))
            time.sleep(1)
        print('child process end, pid={0}, ppid={1}'.format(os.getpid(), os.getppid()))


if __name__ == '__main__':
    main_process_time = 5
    child_process_time = 10
    action = MainProcess(main_process_time, child_process_time)
    action.excutor()