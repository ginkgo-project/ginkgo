![Ginkgo](/assets/logo.png)

Ginkgo is a high-performance linear algebra library for manycore systems, with a
focus on sparse solution of linear systems. It is implemented using modern C++
(you will need at least C++11 compliant compiler to build it), with GPU kernels
implemented in CUDA.


Performance
-----------

An extensive database of up-to-date benchmark results is available in the
[performance data repository](https://github.com/ginkgo-project/ginkgo-data).
Visualizations of the database can be interactively generated using the
[Ginkgo Performance Explorer web application](https://ginkgo-project.github.io/gpe).
The benchmark results are automatically updated using the CI system to always
reflect the current state of the library.

Prerequisites
-------------


### Linux

For Ginkgo core library:

*   _cmake 3.1+_
*   C++11 compliant compiler, one of:
    *   _gcc 5.4.0+_
    *   _clang 3.3+_ (__TODO__: verify, works with 5.0)

The Ginkgo CUDA module has the following __additional__ requirements:

*   _cmake 3.10+_
*   _CUDA 7.0+_ (__TODO__: verify, works with 8.0)
*   Any host compiler restrictions your version of CUDA may impose also apply
    here. For the newest CUDA version, this information can be found in the
    [CUDA installation guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements)

In addition, if you want to contribute code to Ginkgo, you will also need the
following:

*   _clang-format 5.0.1+_ (ships as part of _clang_)


### Mac OS

For Ginkgo core library:

*   _cmake 3.1+_
*   C++11 compliant compiler, one of:
    *   _gcc 5.4.0+_ (__TODO__: verify)
    *   _clang 3.3+_ (__TODO__: verify)
    *   _Apple LLVM 8.0+_ (__TODO__: verify)

The Ginkgo CUDA module has the following __additional__ requirements:

*   _cmake 3.8+_
*   _CUDA 7.0+_ (__TODO__: verify)
*   Any host compiler restrictions your version of CUDA may impose also apply
    here. For the newest CUDA version, this information can be found in the
    [CUDA installation guide](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html)

In addition, if you want to contribute code to Ginkgo, you will also need the
following:

*   _clang-format 5.0.1+_ (ships as part of _clang_)
*   __NOTE:__ If you want to use _clang_ as your compiler and develop Ginkgo,
    you'll currently need two versions _clang_: _clang 4.0.0_ or older, as this
    is this version supporetd by the CUDA 9.1 toolkit, and _clang 5.0.1_ or
    newer, which will not be used for compilation, but only provide the
    _clang-format_ utility


### Windows

Windows is currently not supported, but we are working on porting the library
there. If you are interested in helping us with this effort, feel free to
contact one of the developers. (The library itself doesn't use any non-standard
C++ features, so most of the effort here is in modifying the build system.)

__TODO:__ Some restrictions will also apply on the version of C and C++ standard
libraries installed on the system. We need to investigate this further.


Installation
------------

Use the standard cmake build procedure:

```sh
mkdir build; cd build
cmake -G "Unix Makefiles" [OPTIONS] .. && make
```

Replace `[OPTIONS]` with desired cmake options for your build.
Ginkgo adds the following additional switches to control what is being built:

*   `-DDEVEL_TOOLS={ON, OFF}` sets up the build system for development
    (requires clang-format, will also download git-cmake-format),
    default is `ON`
*   `-DBUILD_TESTS={ON, OFF}` builds Ginkgo's tests
    (will download googletest), default is `ON`
*   `-DBUILD_BENCHMARKS={ON, OFF}` builds Ginkgo's benchmarks
    (will download gflags and rapidjson), default is `ON`
*   `-DBUILD_EXAMPLES={ON, OFF}` builds Ginkgo's examples, default is `ON`
*   `-DBUILD_REFERENCE={ON, OFF}` build reference implementations of the
    kernels, usefull for testing, default os `OFF`
*   `-DBUILD_OMP={ON, OFF}` builds optimized OpenMP versions of the kernels,
    default is `OFF`
*   `-DBUILD_CUDA={ON, OFF}` builds optimized cuda versions of the kernels
    (requires CUDA), default is `OFF`
*   `-DBUILD_DOC={ON, OFF}` creates an HTML version of Ginkgo's documentation
    from inline comments in the code
*   `-DSET_CUDA_HOST_COMPILER={ON, OFF}` instructs the build system to
    explicitly set CUDA's host compiler to match the commpiler used to build the
    the rest of the library (otherwise the nvcc toolchain uses its default host
    compiler). Setting this option may help if you're experiencing linking
    errors due to ABI incompatibilities. The default is `OFF`.
*   `-DCMAKE_INSTALL_PREFIX=path` sets the installation path for `make install`.
    The default value is usually something like `/usr/local`
*   `-DCUDA_ARCHITECTURES=<list>` where `<list>` is a semicolon (`;`) separated
    list of architectures. Supported values are:

    *   `Auto`
    *   `Kepler`, `Maxwell`, `Pascal`, `Volta`
    *   `CODE`, `CODE(COMPUTE)`, `(COMPUTE)`

    `Auto` will automatically detect the present CUDA-enabled GPU architectures
    in the system. `Kepler`, `Maxwell`, `Pascal` and `Volta` will add flags for
    all architectures of that particular NVIDIA GPU generation. `COMPUTE` and
    `CODE` are placeholders that should be replaced with compute and code
    numbers (e.g.  for `compute_70` and `sm_70` `COMPUTE` and `CODE` should be
    replaced with `70`. Default is `Auto`.  For a more detailed explanation of
    this option see the
    [`ARCHITECTURES` specification list](https://github.com/ginkgo-project/CudaArchitectureSelector/blob/master/CudaArchitectureSelector.cmake#L58)
    section in the documentation of the CudaArchitectureSelector CMake module.

For example, to build everything (in debug mode), use:

```cmake
mkdir build; cd build
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug -DDEVEL_TOOLS=ON \
      -DBUILD_TESTS=ON -DBUILD_REFERENCE=ON -DBUILD_OMP=ON -DBUILD_CUDA=ON  ..
make
```

__NOTE:__ Currently, the only verified CMake generator is `Unix Makefiles`.
Other generators may work, but are not officially supported.

### Running the unit tests

You need to compile ginkgo with `-DBUILD_TESTS=ON` option to be able to run the
tests. Use the following command inside the build folder to run all tests:

```sh
make test
```

The output should contain several lines of the form:

```
     Start  1: path/to/test
 1/13 Test  #1: path/to/test .............................   Passed    0.01 sec
```

To run only a specific test and see more details results (e.g. if a test failed)
run the following from the build folder:

```sh
./path/to/test
```

where `path/to/test` is the path returned by `make test`.

### Running the benchmarks

In addition to the unit tests designed to verify correctness, Ginkgo also
includes a benchmark suite for checking its performance on the system. To
compile the benchmarks, the flag `-DBUILD_BENCHMARKS=ON` has to be set during
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

A combinaion of the above approaches is also possible (e.g. it may be useful to
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
                benchmark.
    *   `preconditioner` - Runs the preconditioner benchmarks on artificially
                generated block-diagonal matrices.
*   `DRY_RUN={true, false}` - If set to `true`, prepares the system for the
    benchmark runs (downloads the collections, creates the result structure,
    etc.) and outputs the list of commands that would normally be run, but does
    not run the benchmarks themselves. Default is `false`.
*   `EXECUTOR={reference,cuda,omp}` - The executor used for running the
    benchmarks. Default is `cuda`.
*   `SEGMENTS=<N>` - Splits the benchmark suite into `<N>` segments. This option
    is usefull for running the benchmarks on an HPC system with a batch
    scheduler, as it enables partitioning of the benchmark suite and runing it
    concurrently on multiple nodes of the system. If specified, `SEGMENT_ID`
    also has to be set. Default is `1`.
*   `SEGMENT_ID=<I>` - used in combination with the `SEGMENTS` variable. `<I>`
    should be an integer between 1 and `<N>`. If specified, only the `<I>`-th
    segment of the benchmark suite will be run. Default is `1`.
*   `SYSTEM_NAME=<name>` - the name of the system where the benchmarks are being
    run. This option only changes the directory where the benchmark results are
    stored. It can be used to avoid overwritting the benchmarks if multiple
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

### Installing Ginkgo

To install Ginkgo into the specified folder, execute the following command in
the build folder

```sh
make install
```

If the installation prefix (see `CMAKE_INSTALL_PREFIX`) is not writable for your
user, e.g. when installing Ginkgo system-wide, it might be necessary to prefix
the call with `sudo`.

### Licensing

Refer to [ABOUT-LICENSING.md](ABOUT-LICENSING.md) for details.
