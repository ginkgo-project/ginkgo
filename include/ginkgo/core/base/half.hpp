#ifndef GKO_BASE_HALF_HPP_
#define GKO_BASE_HALF_HPP_
#include <complex>
#include <type_traits>


#ifdef __CUDA_ARCH__


#include <cuda_fp16.h>


#elif defined(__HIP_DEVICE_COMPILE__)


#include <hip/hip_fp16.h>


#endif  // __CUDA_ARCH__


namespace gko {}

#endif  // GKO_BASE_HALF_HPP_
