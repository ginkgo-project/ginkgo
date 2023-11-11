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

factor = 0.9/2
width = 5.90666
height = 5.90666
matplotlib.rcParams['figure.figsize'] = (width, height)  # Use the values from \printsizes
matplotlib.rc('legend',fontsize=8) # using a size in points

executor=sys.argv[0]

if executor == 'cuda':
    rec='2'
    tile='16'
elif executor == 'hip':
    rec='2'
    tile='16'
else:
    executor = 'cuda'
    rec='2'
    tile='16'

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
    data = np.genfromtxt('data/' + executor + '_batch_' + solver +'_' +rec + 'rec_' + tile + 'tile' + '.csv', delimiter=',',
                              skip_header=1, names=['batch_size', 'nrows',
                                                    'time','p_time'])
    df = pd.DataFrame(data, columns=['batch_size', 'nrows', 'time','p_time'])
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

cuda_df1 = collect_solver_data('1row', executor, legend_array)
cuda_df2 = collect_solver_data('2row', executor, legend_array)
cuda_df3 = collect_solver_data('vendor', executor, legend_array)

row_list=[64, 128, 512]
batch_list=[16, 64, 128, 256, 1024, 2048, 4096, 16384, 65536]

cuda_df1 = cuda_df1[~cuda_df1["batch_size"].isin(batch_list)]
cuda_df1 = cuda_df1[~cuda_df1["nrows"].isin(row_list)]
cuda_df2 = cuda_df2[~cuda_df2["batch_size"].isin(batch_list)]
cuda_df2 = cuda_df2[~cuda_df2["nrows"].isin(row_list)]
cuda_df3 = cuda_df3[~cuda_df3["batch_size"].isin(batch_list)]
cuda_df3 = cuda_df3[~cuda_df3["nrows"].isin(row_list)]


speedup_df = cuda_df3
speedup_df['vendor-time'] = cuda_df3['time']
speedup_df['vendor-total-time'] = cuda_df3['p_time']+ cuda_df3['time']
speedup_df['ginkgo1-time'] = cuda_df1['time']
speedup_df['ginkgo2-time'] = cuda_df2['time']
# pd.merge([idf, edf])
speedup_df['speedup-total'] = speedup_df['vendor-total-time']/speedup_df['ginkgo1-time']
speedup_df['speedup1'] = speedup_df['vendor-time']/speedup_df['ginkgo1-time']
speedup_df['speedup2'] = speedup_df['vendor-time']/speedup_df['ginkgo2-time']
# print(speedup_df)

#row_list=[16, 32, 256, 1024, 2048]

print(speedup_df)

fig, ax = plt.subplots()
cycler = plt.cycler(color=opts['colorlist'], marker=opts['marklist'])
ax.set_prop_cycle(cycler)

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": "Helvetica",
# })
# pd.options.display.mpl_style = cmap
# speedup_df.set_index("batch_size", inplace=True)
# speedup_df.groupby("nrows")["speedup-total"].plot(legend=True, xlabel="Num batch entries",
#                                             marker="x", markersize=5,ylabel="Speedup")
# speedup_df.groupby("nrows").apply(lambda x:32^x)["speedup-total"].plot(legend=True, xlabel="Num batch entries",
#                                                   ylabel="Speedup", kind='bar')


# speedup_df.group.plot(y="speedup-total", x='batch_size',kind='bar')
cuda_df1.set_index("batch_size", inplace=True)
cuda_df1.groupby("nrows")["time"].plot(legend=False, xlabel="Num batch entries",
                                      marker="x", markersize=5, ylabel="Time(s)")
cuda_df2.set_index("batch_size", inplace=True)
cuda_df2.groupby("nrows")["time"].plot(legend=False, xlabel="Num batch entries",
                                     marker="o", markersize=5, ylabel="Time(s)")
cuda_df3.set_index("batch_size", inplace=True)
cuda_df3.groupby("nrows")["time"].plot(legend=False, xlabel="Num batch entries",
                                     marker="v", markersize=5, ylabel="Time(s)")
# plt.gcf().set_size_inches(width * factor, height * factor)
plt.tight_layout()
plt.yscale("log")
plt.xscale("log")
#plt.legend(legend_array)
#
f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]

handles = [f("s", opts["colorlist"][i]) for i in range(5)]
handles += [f(opts["marklist"][i], "k") for i in range(3)]

labels = ["16", "32", "256", "1024", "2048"] + ["1row", "2row", "vendor"]
plt.legend(handles, labels, loc=2, framealpha=1)

plt.grid(True)

fname = 'tridiag_batch_bar.pdf'

plt.savefig(fname)
# shutil.copy('./'+fname, '/home/pratik/Documents/10MyPapers/2022-thesis/images/batched-solvers/'+fname)
# plt.show()
