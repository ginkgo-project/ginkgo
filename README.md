  <p align="center"><img src="/assets/logo.png" alt="Ginkgo" width="60%" height="60%"></p>

<div align="center">

[![License](https://img.shields.io/github/license/ginkgo-project/ginkgo.svg)](./LICENSE)|[![c++ standard](https://img.shields.io/badge/c%2B%2B-14-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B#Standardization)|[![Documentation](https://img.shields.io/badge/Documentation-latest-blue.svg)](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/)|[![DOI](https://joss.theoj.org/papers/10.21105/joss.02260/status.svg)](https://doi.org/10.21105/joss.02260)
|:-:|:-:|:-:|:-:|


[![Build status](https://gitlab.com/ginkgo-project/ginkgo-public-ci/badges/develop/pipeline.svg)](https://gitlab.com/ginkgo-project/ginkgo-public-ci/-/pipelines?page=1&scope=branches&ref=develop)|[![OSX-build](https://github.com/ginkgo-project/ginkgo/actions/workflows/osx.yml/badge.svg)](https://github.com/ginkgo-project/ginkgo/actions/workflows/osx.yml)|[![Windows-build](https://github.com/ginkgo-project/ginkgo/actions/workflows/windows-msvc-ref.yml/badge.svg)](https://github.com/ginkgo-project/ginkgo/actions/workflows/windows-msvc-ref.yml)
|:-:|:-:|:-:|


[![codecov](https://codecov.io/gh/ginkgo-project/ginkgo/branch/develop/graph/badge.svg)](https://codecov.io/gh/ginkgo-project/ginkgo)|[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=ginkgo-project_ginkgo&metric=sqale_rating)](https://sonarcloud.io/dashboard?id=ginkgo-project_ginkgo)|[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=ginkgo-project_ginkgo&metric=reliability_rating)](https://sonarcloud.io/dashboard?id=ginkgo-project_ginkgo)|[![CDash dashboard](https://img.shields.io/badge/CDash-Access-blue.svg)](https://my.cdash.org/index.php?project=Ginkgo+Project)
|:-:|:-:|:-:|:-:|

</div>


Ginkgo is a high-performance numerical linear algebra library for many-core systems, with a
focus on solution of sparse linear systems. It is implemented using modern C++
(you will need an at least C++14 compliant compiler to build it), with GPU kernels
implemented for NVIDIA, AMD and Intel GPUs.

---

**[Prerequisites](#prerequisites)** |
**[Building Ginkgo](#building-and-installing-ginkgo)** |
**[Tests, Examples, Benchmarks](#tests-examples-and-benchmarks)** |
**[Bug reports](#bug-reports-and-support)** |
**[Licensing](#licensing)** |
**[Contributing](#contributing-to-ginkgo)** |
**[Citing](#citing-ginkgo)**


# Prerequisites

### Linux and Mac OS

For Ginkgo core library:

*   _cmake 3.16+_
*   C++14 compliant compiler, one of:
    *   _gcc 5.5+_
    *   _clang 3.9+_
    *   _Intel compiler 2019+_
    *   _Apple Clang 14.0_ is tested. Earlier versions might also work.
    *   _Cray Compiler 14.0.1+_
    *   _NVHPC Compiler 22.7+_

The Ginkgo CUDA module has the following __additional__ requirements:

*   _cmake 3.18+_ (If CUDA was installed through the NVIDIA HPC Toolkit, we require _cmake 3.22+_)
*   _CUDA 10.1+_ or _NVHPC Package 22.7+_
*   Any host compiler restrictions your version of CUDA may impose also apply
    here. For the newest CUDA version, this information can be found in the
    [CUDA installation guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
    or [CUDA installation guide for Mac Os X](https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html)

The Ginkgo HIP module has the following __additional__ requirements:

* _ROCm 4.5+_
*    the HIP, hipBLAS, hipSPARSE, hip/rocRAND and rocThrust packages compiled with either:
    * _AMD_ backend (using the `clang` compiler)
    * _10.1 <= CUDA < 11_ backend
* if the hipFFT package is available, it is used to implement the FFT LinOps.

The Ginkgo DPC++(SYCL) module has the following __additional__ requirements:

* _oneAPI 2022.1+_
* Set `dpcpp` or `icpx` as the `CMAKE_CXX_COMPILER`
* `c++17` is used to compile Ginkgo
* The following oneAPI packages should be available:
    * oneMKL
    * oneDPL

The Ginkgo MPI module has the following __additional__ requirements:

* MPI 3.1+, ideally GPU-Aware, for best performance

In addition, if you want to contribute code to Ginkgo, you will also need the
following:

*   _clang-format 8.0.0+_ (ships as part of _clang_)
*   _clang-tidy_ (optional, when setting the flag `-DGINKGO_WITH_CLANG_TIDY=ON`)
*   _iwyu_ (Include What You Use, optional, when setting the flag `-DGINKGO_WITH_IWYU=ON`)

### Windows

*   _cmake 3.13+_
*   C++14 compliant 64-bit compiler:
    *   _MinGW : gcc 5.5+_
    *   _Microsoft Visual Studio : VS 2019+_

The Ginkgo CUDA module has the following __additional__ requirements:

*   _CUDA 10.1+_
*   _Microsoft Visual Studio_
*   Any host compiler restrictions your version of CUDA may impose also apply
    here. For the newest CUDA version, this information can be found in the
    [CUDA installation guide for Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

The Ginkgo OMP module has the following __additional__ requirements:
*  _MinGW_

In these environments, two problems can be encountered, the solution for which is described in the
[windows section in INSTALL.md](INSTALL.md#building-ginkgo-in-windows):
* `ld: error: export ordinal too large` needs the compilation flag `-O1`
* `cc1plus.exe: out of memory allocating 65536 bytes` requires a modification of the environment

__NOTE:__ Some restrictions will also apply on the version of C and C++ standard
libraries installed on the system. This needs further investigation.

# Building and Installing Ginkgo

To build Ginkgo, you can use the standard CMake procedure.

```sh
mkdir build; cd build
cmake -G "Unix Makefiles" .. && cmake --build .
cmake --install .
```

By default, `GINKGO_BUILD_REFERENCE` is enabled. You should be able to run
examples with this executor. By default, Ginkgo tries to enable the relevant
modules depending on your machine environment (present of CUDA, ...). You can
also explicitly compile with the OpenMP, CUDA, HIP or DPC++(SYCL) modules enabled to
run the examples with these executors.

Please refer to the [Installation page](./INSTALL.md) for more details.

Tip: After installation, in your CMake project, Ginkgo can be added with `find_package(Ginkgo)` and the the target that is exported is `Ginkgo::ginkgo`.
An example can be found in the [`test_install`](test/test_install/CMakeLists.txt).

# Tests, Examples and Benchmarks

### Testing

Ginkgo does comprehensive unit tests using Google Tests. These tests are enabled by default and can be disabled if necessary by passing the `-DGINKGO_BUILD_TESTS=NO` to the cmake command. More details about running tests can be found in the [TESTING.md page](./TESTING.md).

### Running examples

Various examples are available for you to understand and play with Ginkgo within the `examples/` directory. They can be compiled by passing the `-DGINKGO_BUILD_EXAMPLES=ON` to the cmake command. Each of the examples have commented code with explanations and this can be found within the [online documentation](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/develop/Examples.html).

### Benchmarking

A unique feature of Ginkgo is the ability to run benchmarks and view your results
with the help of the [Ginkgo Performance Explorer (GPE)](https://ginkgo-project.github.io/gpe/).

More details about this can be found in the [BENCHMARKING.md page](./BENCHMARKING.md)

# Bug reports and Support

If you have any questions about using Ginkgo, please use [Github discussions](https://github.com/ginkgo-project/ginkgo/discussions).

If you would like to request a feature, or have encountered a bug, please [create an issue](https://github.com/ginkgo-project/ginkgo/issues/new). Please be sure to describe your problem and provide as much information as possible.

You can also send an email to [Ginkgo's main email address](mailto:ginkgo.library@gmail.com).

# Licensing

Ginkgo is available under the [3-clause BSD license](LICENSE). All contributions
to the project are added under this license.

Depending on the configuration options used when building Ginkgo, third party
software may be pulled as additional dependencies, which have their own
licensing conditions. Refer to [ABOUT-LICENSING.md](ABOUT-LICENSING.md) for
details.

# Contributing to Ginkgo

We are glad that that you would like to contribute to Ginkgo and we are happy to help you with any questions you may have.

If you are contributing for the first time, please add yourself to the list of external contributors with the following info

``` text
Name Surname <email@domain> Institution(s)
```

#### Declaration

Ginkgo's source is distributed with a BSD-3 clause license. By contributing to Ginkgo and adding yourself to the contributors list, you agree to the following statement (also in [contributors.txt](contributors.txt)):

``` text
I hereby place all my contributions in this codebase under a BSD-3-Clause
license, as specified in the repository's LICENSE file.
```

#### Contribution Guidelines

When contributing to Ginkgo, to ease the review process, please follow the guidelines mentioned in [CONTRIBUTING.md](CONTRIBUTING.md).

It also contains other general recommendations such as writing proper commit messages, understanding Ginkgo's library design, relevant C++ information etc.


# Citing Ginkgo

The main Ginkgo paper describing Ginkgo's purpose, design and interface is
available through the following reference:

``` bibtex
@article{ginkgo-toms-2022,
title = {{Ginkgo: A Modern Linear Operator Algebra Framework for High Performance Computing}},
volume = {48},
copyright = {All rights reserved},
issn = {0098-3500},
shorttitle = {Ginkgo},
url = {https://doi.org/10.1145/3480935},
doi = {10.1145/3480935},
number = {1},
urldate = {2022-02-17},
journal = {ACM Transactions on Mathematical Software},
author = {Anzt, Hartwig and Cojean, Terry and Flegar, Goran and Göbel, Fritz and Grützmacher, Thomas and Nayak, Pratik and Ribizel, Tobias and Tsai, Yuhsiang Mike and Quintana-Ortí, Enrique S.},
month = feb,
year = {2022},
keywords = {ginkgo, healthy software lifecycle, High performance computing, multi-core and manycore architectures},
pages = {2:1--2:33}
}
```

For more information on topical subjects, please refer to the [CITING.md
page](CITING.md).
