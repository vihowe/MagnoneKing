# %%
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

# %%
df = pd.read_csv('./perf_surf.csv', skip_blank_lines=True)
x = list(dict.fromkeys(df['cpu_quota'].tolist()).keys())
y = list(dict.fromkeys(df['mem_quota'].tolist()).keys())
x, y = np.meshgrid(y, x)
qps = np.array(df['qps'])
qps = np.reshape(qps, x.shape)
print(x, y)
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
surf = ax.plot_surface(x, y, qps)
