![Ginkgo](/assets/logo.png)

[![Build status](https://gitlab.com/ginkgo-project/ginkgo-public-ci/badges/develop/build.svg)](https://github.com/ginkgo-project/ginkgo/commits/develop)
[![Coverage](https://codecov.io/gh/ginkgo-project/ginkgo/branch/develop/graph/badge.svg)](https://codecov.io/gh/ginkgo-project/ginkgo)
[![CDash dashboard](https://img.shields.io/badge/CDash-Access-blue.svg)](http://my.cdash.org/index.php?project=Ginkgo+Project)
[![Documentation](https://img.shields.io/badge/Documentation-latest-blue.svg)](https://ginkgo-project.github.io/ginkgo/doc/develop/)
[![License](https://img.shields.io/github/license/ginkgo-project/ginkgo.svg)](LICENSE)
[![c++ standard](https://img.shields.io/badge/c%2B%2B-11-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B#Standardization)


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


### Linux and Mac OS 

For Ginkgo core library:

*   _cmake 3.9+_
*   C++11 compliant compiler, one of:
    *   _gcc 5.3+, 6.3+, 7.3+, 8.1+_
    *   _clang 3.9+_
    *   _Apple LLVM 8.0+_ (__TODO__: verify)

The Ginkgo CUDA module has the following __additional__ requirements:

*   _CUDA 9.0+_
*   Any host compiler restrictions your version of CUDA may impose also apply
    here. For the newest CUDA version, this information can be found in the
    [CUDA installation guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
    or [CUDA installation guide for Mac Os X](https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html)

In addition, if you want to contribute code to Ginkgo, you will also need the
following:

*   _clang-format 5.0.1+_ (ships as part of _clang_)

### Windows

Windows is currently not supported, but we are working on porting the library
there. If you are interested in helping us with this effort, feel free to
contact one of the developers. (The library itself doesn't use any non-standard
C++ features, so most of the effort here is in modifying the build system.)

__TODO:__ Some restrictions will also apply on the version of C and C++ standard
libraries installed on the system. We need to investigate this further.

Contributing to Ginkgo
---------------------------

### Licensing

Ginkgo is available under the [3-clause BSD license](LICENSE). All contributions
to the project are added under this license.

Depending on the configuration options used when building Ginkgo, third party
software may be pulled as additional dependencies, which have their own
licensing conditions. Refer to [ABOUT-LICENSING.md](ABOUT-LICENSING.md) for
details.

### Contributing

When contributing for the first time, please add yourself to the list of
external contributors like in the example below.

#### Contributors
I hereby place all my contributions in this codebase under a BSD-3-Clause
license, as specified in the repository's LICENSE file.

Name Surname <email@domain> Institution(s)

#### Contributing guidelines

Contributing guidelines can be accessed in our Wiki under the [Developer's
Homepage](https://github.com/ginkgo-project/ginkgo/wiki/Developers-Homepage).
This page also contains other information useful to developers, such as writing
proper commit messages, understanding Ginkgo's library design, relevant C++
information, and more.

In general, always refer to this page for developer information. If you think
that anything is missing, feel free to [create an
issue](https://github.com/ginkgo-project/ginkgo/issues/new).  

Quick Install
------------

Use the standard cmake build procedure:

```sh
mkdir build; cd build
cmake -G "Unix Makefiles" .. && make
```

By default, `GINKGO_BUILD_REFERENCE` is enabled. You should be able to run examples with this
executor. You would need to explicitly compile with the OpenMP and CUDA modules enabled
to run with these executors. Please refer to the [Installation page](./INSTALL.md).

After the installation, CMake can find ginkgo with `find_package(Ginkgo)`.
An example can be found in the [`test_install`](test_install/CMakeLists.txt).

### Running the benchmarks

A unique feature of Ginkgo is the ability to run benchmarks and view your results
with the help of the [Ginkgo Performance Explorer (GPE)](https://ginkgo-project.github.io/gpe/). 

More details about this can be found in the [documentation](TODO)

### Licensing

Refer to [ABOUT-LICENSING.md](ABOUT-LICENSING.md) for details.
