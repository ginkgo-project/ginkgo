#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import math
import shutil
from matplotlib.transforms import ScaledTranslation
#from tol_colors import tol_cmap, tol_cset


matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        # Adjust to your LaTex-Engine
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "axes.unicode_minus": False,
    }
)

factor = 1/2
width = 5.90666*1.2
height = 5.90666*1.2
matplotlib.rcParams['figure.figsize'] = (width, height)  # Use the values from \printsizes
# matplotlib.rc('legend',fontsize=4) # using a size in points

thesis_home=os.environ['THESIS_ROOT']

opts = { \
         "marklist" : ['o', 'x', '+', '^', 'v', '<', '>', 'd'],
         "colorlist" : ['mediumblue', 'red', 'forestgreen', 'darkmagenta', 'm', 'c', 'pink','k'],
         "linetype" : ['-', '--', '-.', ':', '--', '-.', '--',':'],
         "linewidth" : 0.75,
         "marksize" : [3,3,3,3,3,3,3],
         "markedgewidth" : 1 \
         }

fname = 'data/spmv_rel.csv'
df = pd.read_csv(fname)

fig, ax = plt.subplots()

df['csr_speedup'] = df['dense_time']/df['csr_time']
df['ell_speedup'] = df['dense_time']/df['ell_time']
df['dense_speedup'] = df['dense_time']/df['dense_time']
plot_vars =['csr_speedup', 'ell_speedup','dense_speedup']

cycler = plt.cycler(linestyle=opts['linetype'], color=opts['colorlist'], marker=opts['marklist'])
ax.set_prop_cycle(cycler)
df.plot(y = plot_vars,ax=ax,
               x = 'mat', xlabel="", ylabel="Relative speedup over BatchDense", kind='bar')

plt.gcf().set_size_inches(width, height* factor)
plt.tight_layout()
plt.yscale('log')
plt.setp(plt.gca().get_xticklabels(),rotation='horizontal')

plot_vars =['BatchCsr', 'BatchEll','BatchDense']
plt.legend(plot_vars)
plt.axhline(y=1, color='k', linestyle='--')
plt.grid(True, linestyle='--', linewidth=0.3)

fname = 'spmv_rel_speedup.pdf'

plt.savefig(fname, bbox_inches='tight')
shutil.copy('./'+fname, thesis_home+'/images/batched-solvers/'+fname)
