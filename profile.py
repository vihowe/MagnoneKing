#!/home/vihowe/anaconda3/bin/python
from typing import overload
import numpy as np
import subprocess
import csv
import time


if __name__ == '__main__':
    cpu_quotas = np.arange(10000, 210000, 10000)
    mem_quotas = np.arange(20, 420, 20)
    task_type = [0, 1, 2, 3]
    with open('perf_surf.csv', 'w+') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['task', 'cpu_quota', 'mem_quota', 'running time', 'overhead'])
        for task in task_type:
            for cpu_quota in cpu_quotas:
                for mem_quota in mem_quotas:
                    t_start = time.perf_counter()
                    s = subprocess.Popen(['docker', 'run', '--rm',
                                        f'--cpu-period=100000',
                                        f'--cpu-quota={cpu_quota}',
                                        '-m', f'{mem_quota}M',
                                        '--memory-swap', '-1',
                                        'vihowe/ciyichang:v2',
                                        'python', 'main_plain_dipolev5_20m_nocali.py',
                                        '--task', f'{task}'],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT)
                    output, _ = s.communicate()
                    total_time = (time.perf_counter() - t_start) * 1000
                    print(output, _)
                    elapsed_time = float(output.decode(encoding='utf-8').split(' ')[-2]) * 225
                    print(elapsed_time, total_time)
                    overhead = total_time - elapsed_time
                    # qps = 1000 / elapsed_time
                    # print(qps)
                    csv_writer.writerow([cpu_quota, mem_quota, elapsed_time, overhead, ])
