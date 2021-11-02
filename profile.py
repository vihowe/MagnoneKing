import numpy as np
import subprocess
import csv


if __name__ == '__main__':
    cpu_quotas = np.arange(10000, 110000, 10000)
    mem_quotas = np.arange(10, 110, 10)
    with open('perf_surf.csv', 'w+') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['cpu_quota', 'mem_quota', 'qps'])
        for cpu_quota in cpu_quotas:
            for mem_quota in mem_quotas:
                s = subprocess.Popen(['docker', 'run', '--rm',
                                      f'--cpu-period=100000',
                                      f'--cpu-quota={cpu_quota}',
                                      '-m', f'{mem_quota}M',
                                      '--memory-swap', '1g',
                                      'vihowe/ciyichang:v2'],
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT)
                output, _ = s.communicate()
                print(output, _)
                elapsed_time = float(output.decode(encoding='utf-8').split(' ')[-2])
                qps = 1000 / elapsed_time
                csv_writer.writerow([cpu_quota, mem_quota, qps])