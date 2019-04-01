Running the benchmarks
----------------------

In addition to the unit tests designed to verify correctness, Ginkgo also
includes a benchmark suite for checking its performance on the system. To
compile the benchmarks, the flag `-DGINKGO_BUILD_BENCHMARKS=ON` has to be set during
the `cmake` step. In addition, the [`ssget` command-line
utility](https://github.com/ginkgo-project/ssget) has to be installed on the
system.

The benchmark suite tests Ginkgo's performance using the [SuiteSparse matrix
collection](https://sparse.tamu.edu/) and artificially generated matrices. The
suite sparse collection will be downloaded automatically when the benchmarks are
run. Please note that the entire collection requires roughly 100GB of disk
storage in its compressed format, and roughly 25GB of additional disk space for
intermediate data (such us uncompressing the archive). Additionally, the
benchmark runs usually take a long time (SpMV benchmarks on the complete
collection take roughly 24h using the K20 GPU), and will stress the system.

The benchmark suite is invoked using the `make benchmark` command in the build
directory. The behavior of the suite can be modified using environment
variables. Assuming the `bash` shell is used, these can either be specified via
the `export` command to persist between multiple runs:

```sh
export VARIABLE="value"
...
make benchmark
```

or specified on the fly, on the same line as the `make benchmark` command:

```sh
env VARIABLE="value" ... make benchmark
```

Since `make` sets any variables passed to it as temporary environment variables,
the following shorthand can also be used:

```sh
make benchmark VARIABLE="value" ...
```

A combination of the above approaches is also possible (e.g. it may be useful to
`export` the `SYSTEM_NAME` variable, and specify the others at every benchmark
run).

Supported environment variables are described in the following list:

*   `BENCHMARK={spmv, solver, preconditioner}` - The benchmark set to run.
    Default is `spmv`.
    *   `spmv` - Runs the sparse matrix-vector product benchmarks on the
                 SuiteSparse collection.
    *   `solver` - Runs the solver benchmarks on the SuiteSparse collection.
                The matrix format is determined by running the `spmv` benchmarks
                first, and using the fastest format determined by that
                benchmark. The maximum number of iterations for the iterative
                solvers is set to 10,000 and the requested residual reduction
                factor to 1e-6.
    *   `preconditioner` - Runs the preconditioner benchmarks on artificially
                generated block-diagonal matrices.
*   `DRY_RUN={true, false}` - If set to `true`, prepares the system for the
    benchmark runs (downloads the collections, creates the result structure,
    etc.) and outputs the list of commands that would normally be run, but does
    not run the benchmarks themselves. Default is `false`.
*   `EXECUTOR={reference,cuda,omp}` - The executor used for running the
    benchmarks. Default is `cuda`.
*   `SEGMENTS=<N>` - Splits the benchmark suite into `<N>` segments. This option
    is useful for running the benchmarks on an HPC system with a batch
    scheduler, as it enables partitioning of the benchmark suite and running it
    concurrently on multiple nodes of the system. If specified, `SEGMENT_ID`
    also has to be set. Default is `1`.
*   `SEGMENT_ID=<I>` - used in combination with the `SEGMENTS` variable. `<I>`
    should be an integer between 1 and `<N>`. If specified, only the `<I>`-th
    segment of the benchmark suite will be run. Default is `1`.
*   `SYSTEM_NAME=<name>` - the name of the system where the benchmarks are being
    run. This option only changes the directory where the benchmark results are
    stored. It can be used to avoid overwriting the benchmarks if multiple
    systems share the same filesystem, or when copying the results between
    systems. Default is `unknown`.

Once `make benchmark` completes, the results can be found in
`<Ginkgo build directory>/benchmark/results/${SYSTEM_NAME}/`. The files are
written in the JSON format, and can be analyzed using any of the data
analysis tools that support JSON. Alternatively, they can be uploaded to an
online repository, and analyzed using Ginkgo's free web tool
[Ginkgo Performance Explorer (GPE)](https://ginkgo-project.github.io/gpe/).
(Make sure to change the "Performance data URL" to your repository if using
GPE.)
