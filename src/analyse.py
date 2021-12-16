# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import bisect
import os


# %%
fig, ax = plt.subplots(3, figsize=(8, 12))
df1 = pd.read_csv('latency_0.log', header=None)
df2 = pd.read_csv('latency_1.log', header=None)
y1 = df1.iloc[:, 0].to_list()
y2 = df2.iloc[:, 0].to_list()
min_d = min(len(y1), len(y2))

y1 = y1[:min_d]
y2 = y2[:min_d]

x = np.arange(0, min_d)
ax[0].plot(x[::30], y1[::30], label='container',linewidth=1.5)
ax[0].plot(x[::30], y2[::30], label='virtual machine', linewidth=0.8)

ax[0].legend(fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# ax[0].set_xlabel('Request ID', fontsize=16)
ax[0].set_ylabel('Latency (s)', fontsize=14)
ax[0].set_title('Requests Latency', fontsize=16)

df1 = pd.read_csv('./simulate/load_latency_0.csv')
df2 = pd.read_csv('./simulate/load_latency_1.csv')
y1 = df1.iloc[:, 1].to_list()
y2 = df2.iloc[:, 1].to_list()
min_d = min(len(y1), len(y2))

y1 = y1[:min_d]
y2 = y2[:min_d]

x = np.arange(0, min_d)
ax[1].plot(x[:], y1[:], label='container',linewidth=1.5)
ax[1].plot(x[:], y2[:], label='virtual machine', linewidth=0.8)

ax[1].legend(fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# ax[1].set_xlabel('Request ID', fontsize=14)
ax[1].set_ylabel('Latency (s)', fontsize=14)
ax[1].set_title('Average Request Latency', fontsize=16)

df = pd.read_csv('./simulate/load_latency_1.csv')
load = df['load']
ax[2].plot(np.arange(len(load)), load)
# ax[2].set_xlabel('time (s)')
ax[2].set_ylabel('Load', fontsize=14)
ax[2].set_title('Load', fontsize=16)
plt.savefig('vm&containet.pdf')

# %%
