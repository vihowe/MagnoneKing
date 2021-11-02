# %%
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot

# %%
df = pd.read_csv('./perf_surf.csv', skip_blank_lines=True)
x = dict.fromkeys(df['cpu_quota'].tolist())
y = dict.fromkeys(df['mem_quota'].tolist())
print(x, y)
