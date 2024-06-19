# System Requirements

## Linux and Mac OS

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
* _cmake 3.21+_

The Ginkgo DPC++(SYCL) module has the following __additional__ requirements:

* _oneAPI 2023.1+_
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

## Windows

*   _cmake 3.16+_
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
[windows section in INSTALL.md](windows.md):
* `ld: error: export ordinal too large` needs the compilation flag `-O1`
* `cc1plus.exe: out of memory allocating 65536 bytes` requires a modification of the environment

__NOTE:__ Some restrictions will also apply on the version of C and C++ standard
libraries installed on the system. This needs further investigation.
