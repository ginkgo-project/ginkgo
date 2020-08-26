Running the benchmarks                         {#benchmarking_ginkgo}
----------------------

In addition to the unit tests designed to verify correctness, Ginkgo also
includes an extensive benchmark suite for checking its performance on all Ginkgo
supported systems. The purpose of Ginkgo's benchmarking suite is to allow easy
and complete reproduction of Ginkgo's performance, and to facilitate performance
debugging as well. Most results published in Ginkgo papers are generated thanks
to this benchmarking suite and are accessible online under the [ginkgo-data
repository](https://github.com/ginkgo-project/ginkgo-data/). These results can
also be used for performance comparison in order to ensure that you get similar
performance as what is published on this repository.

To compile the benchmarks, the flag `-GINKGO_BUILD_BENCHMARKS=ON` has to be set
during the `cmake` step. In addition, the [`ssget` command-line
utility](https://github.com/ginkgo-project/ssget) has to be installed on the
system. The purpose of this file is to explain in detail the capacities of this
benchmarking suite as well as how to properly setup everything.

Here is a short description of the content of this file:
1. Ginkgo setup and best practice guidelines
2. Installing and using the `ssget` tool to fetch the [SuiteSparse
   matrices](https://sparse.tamu.edu/).
3. Benchmarking overview and how to run them in a simple way.
4. How to publish the benchmark results online and use the [Ginkgo Performance
   Explorer (GPE)](https://ginkgo-project.github.io/gpe/) for performance
   analysis (optional).
5. Using the benchmark suite for performance debugging thanks to the loggers.
6. All available benchmark customization options.


### 1: Ginkgo setup and best practice guidelines

Before benchmarking Ginkgo, make sure that you follow the general guidelines in
order to ensure best performance.

1. The code should be compiled in `Release` mode.
2. Make sure the machine has no competing jobs. On a Linux machine multiple
   commands can be used, `last` shows the currently opened sessions, `top` or
   `htop` allows to show the current machine load, and if considering using
   specific GPUs, `nvidia-smi` or `rocm-smi` can be used to check their load.
3. By default, Ginkgo's benchmarks will always do at least one warm-up run. For
   better accuracy, every benchmark is also averaged over 10 runs, except for
   the solver benchmark which are usually fairly long. These parameters can be
   tuned at the command line to either shorten benchmarking time or improve
   benchmarking accuracy.

In addition, the following specific options can be considered:
1. When specifically using the adaptive block jacobi preconditioner, enable
   the `GINKGO_JACOBI_FULL_OPTIMIZATIONS` CMake flag. Be careful that this will
   use much more memory and time for the compilation due to compiler performance
   issues with register optimizations, in particular.
2. The current benchmarking setup also allows to benchmark only the overhead by
   using as either (or for all) preconditioner/spmv/solver, the special
   `overhead` LinOp. If your purpose is to check Ginkgo's overhead, make sure to
   try this mode.

### 2: Using ssget to fetch the matrices

The benchmark suite tests Ginkgo's performance using the [SuiteSparse matrix
collection](https://sparse.tamu.edu/) and artificially generated matrices. The
suite sparse collection will be downloaded automatically when the benchmarks are
run. This is done thanks to the [`ssget` command-line
utility](https://github.com/ginkgo-project/ssget).

To install `ssget`, access the repository and copy the file `ssget` into a
directory present in your `PATH` variable as per the tool's `README.md`
instructions. The tool can be installed either in a global system path or a
local directory such as `$HOME/.local/bin`. After installing the tool, it is
important to review the `ssget` script and configure as needed the variable
`ARCHIVE_LOCATION` on line 39. This is where the matrices will be stored into.

The Ginkgo benchmark can be set to run on only a portion of the SuiteSparse
matrix collection as we will see in the following section. Please note that the
entire collection requires roughly 100GB of disk storage in its compressed
format, and roughly 25GB of additional disk space for intermediate data (such us
uncompressing the archive). Additionally, the benchmark runs usually take a long
time (SpMV benchmarks on the complete collection take roughly 24h using the K20
GPU), and will stress the system.

Before proceeding, it can be useful in order to save time to download the
matrices as preparation. This can be done by using the `ssget -f -i i` command
where `i` is the ID of the matrix to be downloaded. The following loop allows
to download the full SuiteSparse matrix collection:

```bash
for i in $(seq 0 $(ssget -n)); do
    ssget -f -i ${i}
done
```

Note that `ssget` can also be used to query properties of the matrix and filter
the matrices which are downloaded. For example, the following will download only
positive definite matrices with less than 500M non zero elements and 10M
columns. Please refer to the [`ssget`
documentation](https://github.com/ginkgo-project/ssget/blob/master/README.md)
for more information.

```bash
for i in $(seq 0 $(ssget -n)); do
    posdef=$(ssget -p posdef -i ${i})
    cols=$(ssget -p cols -i ${i})
    nnz=$(ssget -p nonzeros -i ${i})
    if [ "$posdef" -eq 1 -a "$cols" -lt 10000000 -a "$nnz" -lt 500000000 ]; then
        ssget -f -i ${i}
    fi
done
```

### 3: Benchmarking overview

The benchmark suite is invoked using the `make benchmark` command in the build
directory. Under the hood, this command simply calls the script
`benchmark/run_all_benchmarks.sh` so it is possible to manually launch this
script as well. The behavior of the suite can be modified using environment
variables. Assuming the `bash` shell is used, these can either be specified via
the `export` command to persist between multiple runs:

```bash
export VARIABLE="value"
...
make benchmark
```

or specified on the fly, on the same line as the `make benchmark` command:

```bash
VARIABLE="value" ... make benchmark
```

Since `make` sets any variables passed to it as temporary environment variables,
the following shorthand can also be used:

```bash
make benchmark VARIABLE="value" ...
```

A combination of the above approaches is also possible (e.g. it may be useful to
`export` the `SYSTEM_NAME` variable, and specify the others at every benchmark
run).

The benchmark suite can take a number of configuration parameters. Benchmarks
can be run only for `sparse matrix vector products (spmv)`, for full solvers
(with or without preconditioners), or for preconditioners only when supported.
The benchmark suite also allows to target a sub-part of the SuiteSparse matrix
collection. For details, see the [available benchmark options](### 5: Available
benchmark options). Here are the most important options:
* `BENCHMARK={spmv, solver, preconditioner}` - allows to select the type of
    benchmark to be ran.
* `EXECUTOR={reference,cuda,hip,omp}` - select the executor and platform the
    benchmarks should be ran on.
* `SYSTEM_NAME=<name>` - a name which will be used to designate this platform
    (e.g. V100, RadeonVII, ...).
* `SEGMENTS=<N>` - Split the benchmarked matrix space into `<N>` segments. If
    specified, `SEGMENT_ID` also has to be set.
* `SEGMENT_ID=<I>` - used in combination with the `SEGMENTS` variable. `<I>`
    should be an integer between 1 and `<N>`, the number of `SEGMENTS`. If
    specified, only the `<I>`-th segment of the benchmark suite will be run.
* `MATRIX_LIST_FILE=/path/to/matrix_list.file` - allows to list SuiteSparse
    matrix id or name to benchmark. As an example, a matrix list file containing
    the following will ensure that benchmarks are ran for only those three
    matrices:

    ```
    1903
    Freescale/circuit5M
    thermal2
    ```

### 4: Publishing the results on Github and analyze the results with the GPE (optional)

The previous experiments generated json files for each matrices, each containing
timing, iteration count, achieved precision, ... depending on the type of
benchmark run. These files are available in the directory
`${ginkgo_build_dir}/benchmark/results/`. These files can be analyzed and
processed through any tool (e.g. python). In this section, we describe how to
generate the plots by using Ginkgo's
[GPE](https://ginkgo-project.github.io/gpe/) tool. First, we need to publish the
experiments into a Github repository which will be then linked as source input
to the GPE. For this, we can simply fork the ginkgo-data repository. To do so,
we can go to the github repository and use the forking interface:
https://github.com/ginkgo-project/ginkgo-data/

Once it's done, we want to clone the repository locally, put all
results online and access the GPE for plotting the results. Here are the
detailed steps:

```bash
git clone https://github.com/<username>/ginkgo-data.git $HOME/ginkgo_benchmark/ginkgo-data
# send the benchmarked data to the ginkgo-data repository
# If needed, remove the old data so that no previous data is left.
# rm -r ${HOME}/ginkgo_benchmark/ginkgo-data/data/${SYSTEM_NAME}
rsync -rtv ${ginkgo_build_dir}/benchmark/results/ $HOME/ginkgo_benchmark/ginkgo-data/data/
cd ${HOME}/ginkgo_benchmark/ginkgo-data/data/
# The following updates the main `.json` files with the list of data.
# Ensure a python 3 installation is available.
./build-list . > list.json
./agregate < list.json > agregate.json
./represent . > represent.json
git config --local user.name "<Name>"
git config --local user.email "<email>"
git commit -am "Ginkgo benchmark ${BENCHMARK} of ${SYSTEM_NAME}..."
git push
```

Note that depending on what data is of interest, you may need to update the
scripts `build-list` or `agregate` to change which files you want to agglomerate
and summarize (depending on the system name), or which data you want to select
(solver results, spmv results, ...).

For the generating the plots in the GPE, here are the steps to go through:
1. Access the GPE: https://ginkgo-project.github.io/gpe/
2. Update data root URL, from
   `https://raw.githubusercontent.com/ginkgo-project/ginkgo-data/master/data` to
   `https://raw.githubusercontent.com/<username>/ginkgo-data/<branch>/data`
3. Click on the arrow to load the data, select the `Result Summary` entry above.
4. Click on `select an example` to choose a plotting script. Multiple scripts
   are available by default in different branches. You can use the `jsonata` and
   `chartjs` languages to develop your own as well.
5. The results should be available in the tab "plot" on the right side. Other
   tabs allow to access the result of the processed data after invoking the
   processing script.

### 5: Detailed performance analysis and debugging

Detailed performance analysis can be ran by passing the environment variable
`DETAILED=1` to the benchmarking script. This detailed run is available for
solvers and allows to log the internal residual after every iteration as well as
log the time taken by all operations. These features are also available in the
`performance-debugging` example which can be used instead and modified as needed
to analyze Ginkgo's performance.

These features are implemented thanks to the loggers located in the file
`${ginkgo_src_dir}/benchmark/utils/loggers.hpp`. Ginkgo possesses hooks at all important code
location points which can be inspected thanks to the logger. In this fashion, it
is easy to use these loggers also for tracking memory allocation sizes and other
important library aspects.

### 6: Available benchmark options

There are a set amount of options available for benchmarking. Most important
options can be configured through the benchmarking script itself thanks to
environment variables. Otherwise, some specific options are not available
through the benchmarking scripts but can be directly configured when running the
benchmarking program itself. For a list of all options, run for example
`${ginkgo_build_dir}/benchmark/solver/solver --help`.

The supported environment variables are described in the following list:
* `BENCHMARK={spmv, solver, preconditioner}` - allows to select the type of
    benchmark to be ran. Default is `spmv`.
    *   `spmv` - Runs the sparse matrix-vector product benchmarks on the
                 SuiteSparse collection.
    *   `solver` - Runs the solver benchmarks on the SuiteSparse collection. The
                 matrix format is determined by running the `spmv` benchmarks
                 first, and using the fastest format determined by that
                 benchmark.
    *   `preconditioner` - Runs the preconditioner benchmarks on artificially
                 generated block-diagonal matrices.
* `EXECUTOR={reference,cuda,hip,omp}` - select the executor and platform the
    benchmarks should be ran on. Default is `cuda`.
* `SYSTEM_NAME=<name>` - a name which will be used to designate this platform
    (e.g. V100, RadeonVII, ...) and not overwrite previous results. Default is
    `unknown`.
* `SEGMENTS=<N>` - Split the benchmarked matrix space into `<N>` segments. If
    specified, `SEGMENT_ID` also has to be set.  Default is `1`.
* `SEGMENT_ID=<I>` - used in combination with the `SEGMENTS` variable. `<I>`
    should be an integer between 1 and `<N>`, the number of `SEGMENTS`. If
    specified, only the `<I>`-th segment of the benchmark suite will be run.
    Default is `1`.
* `MATRIX_LIST_FILE=/path/to/matrix_list.file` - allows to list SuiteSparse
    matrix id or name to benchmark. As an example, a matrix list file containing
    the following will ensure that benchmarks are ran for only those three
    matrices:
    ```
    1903
    Freescale/circuit5M
    thermal2
    ```
* `DEVICE_ID` - the accelerator device ID to target for the benchmark. The
    default is `0`.
* `DRY_RUN={true, false}` - If set to `true`, prepares the system for the
    benchmark runs (downloads the collections, creates the result structure,
    etc.) and outputs the list of commands that would normally be run, but does
    not run the benchmarks themselves. Default is `false`.
* `PRECONDS={jacobi,adaptive-jacobi,ilu,parict,parilu,parilut,none}` the
    preconditioners to use for either `solver` or `preconditioner` benchmarks.
    Multiple options can be passed to this variable. Default is `none`.
* `FORMATS={csr,coo,ell,hybrid,sellp,hybridxx,cusp_xx,hipsp_xx}` the matrix
    formats to benchmark for the `spmv` phase of the benchmark. Run
    `${ginkgo_build_dir}/benchmark/spmv/spmv --help` for a full list. If needed,
    multiple options for hybrid with different optimization parameters are
    available. Depending on the libraries available at build time, vendor
    library formats (cuSPARSE with `cusp_` prefix or hipSPARSE with `hipsp_`
    prefix) can be used as well. Multiple options can be passed. The default is
    `csr,coo,ell,hybrid,sellp`.
* `SOLVERS={bicgstab,bicg,cg,cgs,fcg,gmres}` - the solvers which should be
    benchmarked. Multiple options can be passed. The default is `cg`.
* `SOLVERS_PRECISION=<precision>` - the minimal residual reduction before which
    the solver should stop. The default is `1e-6`.
* `SOLVERS_MAX_ITERATION=<number>` - the maximum number of iterations with which
    a solver should be ran. The default is `10000`.
* `DETAILED={0,1}` - selects whether detailed benchmarks should be ran for the
    solver benchmarks, can be either `0` (off) or `1` (on). The default is `0`.
