#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import sys
import shutil

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

factor = 0.6
width = 5.90666
height = 5.90666
matplotlib.rcParams['figure.figsize'] = (width, height)  # Use the values from \printsizes
matplotlib.rc('legend',fontsize=8) # using a size in points

executor=sys.argv[1]

suffix='_131072b_'

if executor == 'cuda':
    rec='2'
    tile='16'
elif executor == 'hip':
    rec='4'
    tile='32'

opts = { \
         "marklist" : ['o', 'x', 'v', '+', '^'],
         "colorlist" : ['mediumblue', 'red', 'forestgreen', 'c', 'm'],
         "linetype" : ['-', '--', '-.', ':', '--'],
         "linewidth" : 0.75,
         "marksize" : [3,3,3,3,3],
         "markedgewidth" : 1 \
         }

def collect_solver_data(solver, executor, legend_array):
    """
    """
    #fname='data/' + executor + '_batch_' + solver +'_' +rec + 'rec_' + tile + 'tile' + '.csv'
    fname = 'data/' + executor + '_batch_' + solver + suffix + '.csv'
    data = np.genfromtxt(fname, delimiter=',',
                              skip_header=1, names=['nrows', 'nrec','tile_size',
                                                    'time','p_time'],filling_values = 0,invalid_raise=False)
    df = pd.DataFrame(data, columns=['nrows', 'nrec','tile_size', 'time','p_time'])
    #
    if executor == 'omp':
        exec_string = 'omp(76)'
    elif executor == 'cuda':
        exec_string = 'a100'
    elif executor == 'hip':
        exec_string = 'mi250x'
    else:
        exec_string = executor

    if solver == '1row':
        sol_string = '-ginkgo-1row'
    elif solver == '2row':
        sol_string = '-ginkgo-2row'
    else:
        sol_string = '-vendor'

    legend_array.extend([exec_string + sol_string])

    return df


legend_array = list()

df1 = collect_solver_data('1row', executor, legend_array)
# df2 = collect_solver_data('2row', executor, legend_array)
# df3 = collect_solver_data('vendor', executor, legend_array)

print(df1)

speedup_df = df3
speedup_df['vendor-time'] = df3['time']
speedup_df['vendor-total-time'] = df3['p_time']+ df3['time']
speedup_df['ginkgo1-time'] = df1['time']
speedup_df['ginkgo2-time'] = df2['time']
# pd.merge([idf, edf])
speedup_df['speedup-total'] = speedup_df['vendor-total-time']/speedup_df['ginkgo1-time']
speedup_df['speedup1'] = speedup_df['vendor-time']/speedup_df['ginkgo1-time']
speedup_df['speedup2'] = speedup_df['vendor-time']/speedup_df['ginkgo2-time']
# print(speedup_df)

batch_sizes=[32,512,8192 ,32768, 1317092]
# batch_sizes = [str(r) for r in batch_sizes]

print(speedup_df)

strat_type='speedup1'

r32 = [speedup_df.loc[speedup_df.batch_size==32 ,strat_type].values[0],
       speedup_df.loc[speedup_df.batch_size==512 ,strat_type].values[0],
       speedup_df.loc[speedup_df.batch_size==8192 ,strat_type].values[0],
       speedup_df.loc[speedup_df.batch_size==32768 ,strat_type].values[0],
       speedup_df.loc[speedup_df.batch_size==131072,strat_type].values[0],
       ]
r256 = [speedup_df.loc[speedup_df.batch_size==32 ,strat_type].values[1],
       speedup_df.loc[speedup_df.batch_size==512 ,strat_type].values[1],
       speedup_df.loc[speedup_df.batch_size==8192,strat_type].values[1],
       speedup_df.loc[speedup_df.batch_size==32768,strat_type].values[1],
       speedup_df.loc[speedup_df.batch_size==131072,strat_type].values[1]]
r2048 = [speedup_df.loc[speedup_df.batch_size==32 ,strat_type].values[2],
       speedup_df.loc[speedup_df.batch_size==512 ,strat_type].values[2],
       speedup_df.loc[speedup_df.batch_size==8192,strat_type].values[2],
       speedup_df.loc[speedup_df.batch_size==32768,strat_type].values[2],
       speedup_df.loc[speedup_df.batch_size==131072,strat_type].values[2]]

df = pd.DataFrame(data={'32': r32, '256': r256, '2048': r2048}, index=batch_sizes)
print(df)

# df.index = df.index.batch_sizes

print(df)

fig, ax = plt.subplots()
cycler = plt.cycler(color=opts['colorlist'], marker=opts['marklist'])
ax.set_prop_cycle(cycler)

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": "Helvetica",
# })
# pd.options.display.mpl_style = cmap

df.plot(kind='bar', rot=45, xlabel='Num batch entries', ylabel='Speedup')
# # add some labels
# for c in ax.containers:
#     # set the bar label
#     ax.bar_label(c, fmt='%.0f', label_type='edge')


# speedup_df.set_index("batch_size", inplace=True)
# speedup_df["speedup1"].plot(legend=True, xlabel="Num batch entries",
#                                                   ylabel="Speedup", kind='bar')
# plt.setp(plt.gca().get_xticklabels(),rotation='horizontal')
# speedup_df["speedup2"].plot(legend=True, xlabel="Num batch entries",
#                                                   ylabel="Speedup", kind='bar')
# speedup_df.groupby("nrows").apply(lambda x:32^x)["speedup-total"].plot(legend=True, xlabel="Num batch entries",
#                                                   ylabel="Speedup", kind='bar')


# speedup_df.group.plot(y="speedup-total", x='batch_size',kind='bar')
# df1.set_index("batch_size", inplace=True)
# df1.groupby("nrows")["time"].plot(legend=False, xlabel="Num batch entries",
#                                       marker="x", markersize=5, ylabel="Time(s)")
# df2.set_index("batch_size", inplace=True)
# df2.groupby("nrows")["time"].plot(legend=False, xlabel="Num batch entries",
#                                      marker="o", markersize=5, ylabel="Time(s)")
# df3.set_index("batch_size", inplace=True)
# df3.groupby("nrows")["time"].plot(legend=False, xlabel="Num batch entries",
#                                      marker="v", markersize=5, ylabel="Time(s)")
plt.gcf().set_size_inches(width * factor, height * factor)
plt.tight_layout()
plt.axhline(y=1, color='k', linestyle='--')
plt.grid(True, linestyle='--', linewidth=0.3)
# plt.yscale("log")
# plt.xscale("log")
#plt.legend(legend_array)
#
# f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]

# handles = [f("s", opts["colorlist"][i]) for i in range(3)]
# handles += [f(opts["marklist"][i], "k") for i in range(3)]

# labels = ["32", "256", "2048"]
# plt.legend(handles, labels, loc=2, framealpha=1)

plt.grid(True)

fname = 'tridiag_batch_speedup_' + strat_type + executor + '.pdf'

plt.savefig(fname)
# shutil.copy('./'+fname, '/home/pratik/Documents/10MyPapers/2022-thesis/images/batched-solvers/'+fname)
# plt.show()
