# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import bisect
import os


# %%
fig, ax = plt.subplots(figsize=(8,5))
df1 = pd.read_csv('latency_0.log', header=None)
df2 = pd.read_csv('latency_1.log', header=None)
y1 = df1.iloc[:, 0].to_list()
y2 = df2.iloc[:, 0].to_list()
min_d = min(len(y1), len(y2))

y1 = y1[:min_d]
y2 = y2[:min_d]

x = np.arange(0, min_d)
ax.plot(x[::10], y1[::10], label='container',linewidth=1.5)
ax.plot(x[::10], y2[::10], label='virtual machine', linewidth=0.8)

ax.legend(fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.set_xlabel('Request ID', fontsize=16)
ax.set_ylabel('Latency (s)', fontsize=16)
plt.savefig('./latency.pdf', bbox_inches='tight')

# %%
