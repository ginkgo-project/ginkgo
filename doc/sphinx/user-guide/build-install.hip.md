# Building Ginkgo with HIP support

Ginkgo provides a [HIP](https://github.com/ROCm-Developer-Tools/HIP) backend.
This allows to compile optimized versions of the kernels for either AMD or
NVIDIA GPUs. The CMake configuration step will try to auto-detect the presence
of HIP either at `/opt/rocm/hip` or at the path specified by `HIP_PATH` as a
CMake parameter (`-DHIP_PATH=`) or environment variable (`export HIP_PATH=`),
unless `-DGINKGO_BUILD_HIP=ON/OFF` is set explicitly.

## Changing the paths to search for HIP and other packages
All HIP installation paths can be configured through the use of environment
variables or CMake variables. This way of configuring the paths is currently
imposed by the `HIP` tool suite. The variables are the following:
+ CMake `-DROCM_PATH=` or environment `export ROCM_PATH=`: sets the `ROCM`
  installation path. The default value is `/opt/rocm/`.
+ CMake `-DHIP_CLANG_PATH` or environment `export HIP_CLANG_PATH=`: sets the
  `HIP` compatible `clang` binary path. The default value is
  `${ROCM_PATH}/llvm/bin`.
+ CMake `-DHIP_PATH=` or environment `export HIP_PATH=`: sets the `HIP`
  installation path. The default value is `${ROCM_PATH}/hip`.
+ CMake `-DHIPBLAS_PATH=` or environment `export HIPBLAS_PATH=`: sets the
  `hipBLAS` installation path. The default value is `${ROCM_PATH}/hipblas`.
+ CMake `-DHIPSPARSE_PATH=` or environment `export HIPSPARSE_PATH=`: sets the
  `hipSPARSE` installation path. The default value is `${ROCM_PATH}/hipsparse`.
+ CMake `-DHIPFFT_PATH=` or environment `export HIPFFT_PATH=`: sets the
  `hipFFT` installation path. The default value is `${ROCM_PATH}/hipfft`.
+ CMake `-DROCRAND_PATH=` or environment `export ROCRAND_PATH=`: sets the
  `rocRAND` installation path. The default value is `${ROCM_PATH}/rocrand`.
+ CMake `-DHIPRAND_PATH=` or environment `export HIPRAND_PATH=`: sets the
  `hipRAND` installation path. The default value is `${ROCM_PATH}/hiprand`.
+ environment `export CUDA_PATH=`: where `hipcc` can find `CUDA` if it is not in
  the default `/usr/local/cuda` path.


## HIP platform detection of AMD and NVIDIA
Ginkgo relies on CMake to decide which compiler to use for HIP.
To choose `nvcc` instead of the default ROCm `clang++`, set the corresponding
environment variable:
```bash
export HIPCXX=nvcc
```
Note that this option is currently not being tested in our CI pipelines.
