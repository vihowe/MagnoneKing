#!/home/vihowe/anaconda3/bin/python
"""Profile each location task's performance in term of the cpu and memory share

"""
import numpy as np
import subprocess
import csv
import time

if __name__ == '__main__':
    cpu_quotas = np.arange(5000, 120000, 20000)
    mem_quotas = np.arange(30, 150, 30)
    task_type = [0, 1, 2, 3]
    with open('data/2_2.csv', 'w+') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            ['task', 'cpu_quota', 'mem_quota', 'running time', 'overhead'])
        for task in task_type:
            for cpu_quota in cpu_quotas:
                for mem_quota in mem_quotas:
                    print(
                        f"===For task{task}:\tcpu:{cpu_quota/100000}\tmem:{mem_quota}M"
                    )
                    t_start = time.perf_counter()
                    s = subprocess.Popen([
                        'docker', 'run', '--rm', f'--cpu-period=100000',
                        f'--cpu-quota={cpu_quota}', '-m', f'{mem_quota}M',
                        '--memory-swap', '-1', 'vihowe/ciyichang:x86', 'python',
                        'main_plain_dipolev5_20m_nocali.py', '--task',
                        f'{task}'
                    ],
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT)
                    output, _ = s.communicate()
                    total_time = (time.perf_counter() - t_start) * 1000
                    # print(output, _)
                    try:
                        elapsed_time = float(
                            output.decode(encoding='utf-8').split(' ')[-2])
                    except ValueError:
                        print(output)
                    print(f"\trunning time: {elapsed_time}")
                    overhead = total_time - elapsed_time * 25
                    csv_writer.writerow([
                        task,
                        cpu_quota / 100000,
                        mem_quota,
                        elapsed_time,
                        overhead,
                    ])
