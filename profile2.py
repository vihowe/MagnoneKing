#!/home/vihowe/anaconda3/bin/python
import numpy as np
import subprocess
import csv
import time

if __name__ == '__main__':
    cpu_quotas = np.arange(100000, 220000, 100000)
    mem_quotas = np.arange(1, 3, 1)
    # task_type = [0, 1, 2, 3]
    with open('data/r.csv', 'w+') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            ['cpu_quota', 'mem_quota', 'startup_time',])
        for cpu_quota in cpu_quotas:
            for mem_quota in mem_quotas:
                t_start = time.perf_counter()
                s = subprocess.Popen([
                    'docker', 'run', '--rm', f'--cpu-period=100000',
                    f'--cpu-quota={cpu_quota}', '-m', f'{mem_quota}g',
                    '--memory-swap', '-1', 'vihowe/ciyichang:x86'
                ],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT)
                # output, _ = s.communicate()
                total_time = (time.perf_counter() - t_start) * 1000
                # print(output, _)
                print(f"\trunning time: {total_time}")
                csv_writer.writerow([
                    cpu_quota / 100000,
                    mem_quota,
                    total_time
                ])

