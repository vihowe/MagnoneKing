# %%
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
# %%

def draw_perf_surf(file_name):
    
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    data = pd.read_csv(f'{file_name}.csv')
    for task in range(4):
        plt.cla()
        task_data = data.loc[data['task']==task]
        total_time = 1000 / (task_data['running time'] + task_data['overhead'])
        cpu_quota = np.arange(0.2, 2.2, 0.2)
        mem_quota = np.arange(40, 240, 20)
        X, Y = np.meshgrid(mem_quota, cpu_quota)
        total_time = total_time.to_numpy().reshape(10, 10)
        ax.plot_surface(X, Y, total_time, linewidth=0)
        plt.savefig(f'./{file_name}_{task}.pdf')


# %%
draw_perf_surf('4')


# %%
