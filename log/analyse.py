# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import bisect
import os

# %%
dire = './1'


# %%
fig, ax = plt.subplots(figsize=(8,5))
df1 = pd.read_csv(os.path.join(dire, 'latency_affinity.log'), header=None)
df2 = pd.read_csv(os.path.join(dire, 'latency_naive.log'), header=None)
df3 = pd.read_csv(os.path.join(dire, 'latency_random.log'), header=None)
y1 = df1.iloc[:, 0].to_list()
y2 = df2.iloc[:, 0].to_list()
y3 = df3.iloc[:, 0].to_list()
min_d = min(len(y1), len(y2), len(y3))

y1 = y1[:min_d]
y2 = y2[:min_d]
y3 = y3[:min_d]
x = np.arange(0, min_d)
ax.plot(x[::40], y1[::40], label='affinity',linewidth=1.5)
ax.plot(x[::40], y2[::40], label='naive', linewidth=0.8)
ax.plot(x[::40], y3[::40], label='random', linewidth=0.8)

ax.legend(fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.set_xlabel('Request ID', fontsize=16)
ax.set_ylabel('Latency (s)', fontsize=16)
plt.savefig('./latency.pdf', bbox_inches='tight')
# %%

fig, ax = plt.subplots()
y1 = sorted(y1)
y2 = sorted(y2)
y3 = sorted(y3)
min_d = min([y1[0], y2[0], y3[0]])
max_d = max([y1[-1], y2[-1], y3[-1]])
Y1 = []
Y2 = []
Y3 = []
X = np.linspace(min_d, max_d, 50)
for x in X:
    idx1 = bisect.bisect(y1, x)
    idx2 = bisect.bisect(y2, x)
    idx3 = bisect.bisect(y3, x)
    Y1.append(idx1 / len(y1))
    Y2.append(idx2 / len(y2))
    Y3.append(idx3 / len(y3))
ax.plot(X, Y1, label='affinity', linewidth=1.8)
ax.plot(X, Y2, label='naive')
ax.plot(X, Y3, label='random')
# ax.set_xlabel('Request ID', fontsize=16)
ax.set_xlabel('Latency (s)', fontsize=16)

ax.legend(fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('./latency_cdf.pdf', bbox_inches='tight')

# %%
fig, ax = plt.subplots()
df = pd.read_csv('../src/simulate/load.csv', header=None)
load = df.iloc[:, 0]
x = np.arange(0, len(load))
ax.plot(x, load)
ax.set_ylabel('Load (req/s)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('./load.pdf', bbox_inches='tight')
# %%
