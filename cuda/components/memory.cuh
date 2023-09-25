/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CUDA_COMPONENTS_MEMORY_CUH_
#define GKO_CUDA_COMPONENTS_MEMORY_CUH_


#include <type_traits>


#include <ginkgo/core/base/math.hpp>


#include "cuda/base/types.hpp"


namespace gko {
namespace kernels {
namespace cuda {


/**
 * Transforms a generic CUDA pointer pointing to shared memory to a
 * shared memory pointer for use in PTX assembly.
 * CUDA PTX assembly uses 32bit pointers for shared memory addressing.
 * The result is undefined for a generic pointer pointing to anything but
 * shared memory.
 */
__device__ __forceinline__ uint32 convert_generic_ptr_to_smem_ptr(void* ptr)
{
// see
// https://github.com/NVIDIA/cutlass/blob/
//     6fc5008803fe4e81b81a836fcd3a88258f4e5bbf/
//     include/cutlass/arch/memory_sm75.h#L90
// for reasoning behind this implementation
#if (!defined(__clang__) && __CUDACC_VER_MAJOR__ >= 11)
    return static_cast<uint32>(__cvta_generic_to_shared(ptr));
#elif (!defined(__clang__) && CUDACC_VER_MAJOR__ == 10 && \
       __CUDACC_VER_MINOR__ >= 2)
    return __nvvm_get_smem_pointer(ptr);
#else
    uint32 smem_ptr;
    asm("{{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 "
        "%0, smem_ptr; }}"
        : "=r"(smem_ptr)
        : "l"(ptr));
    return smem_ptr;
#endif
}


__device__ __forceinline__ uint32 membar_acq_rel()
{
#if __CUDA_ARCH__ < 700
    asm volatile("membar.gl;" ::: "memory");
#else
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
#endif
}


__device__ __forceinline__ uint32 membar_acq_rel_shared()
{
#if __CUDA_ARCH__ < 700
    asm volatile("membar.cta;" ::: "memory");
#else
    asm volatile("fence.acq_rel.cta;" ::: "memory");
#endif
}


#include "common/cuda_hip/components/memory.hpp.inc"


__device__ __forceinline__ int32 load_relaxed_shared(const int32* ptr)
{
    int32 result;
#if __CUDA_ARCH__ < 700
    asm volatile("ld.volatile.shared.b32 %0, [%1];"
                 : "=r"(result)
                 : "r"(convert_generic_ptr_to_smem_ptr(const_cast<int32*>(ptr)))
                 : "memory");
#else
    asm volatile("ld.relaxed.cta.shared.b32 %0, [%1];"
                 : "=r"(result)
                 : "r"(convert_generic_ptr_to_smem_ptr(const_cast<int32*>(ptr)))
                 : "memory");
#endif

    return result;
}


__device__ __forceinline__ void store_relaxed_shared(int32* ptr, int32 result)
{
#if __CUDA_ARCH__ < 700
    asm volatile("st.volatile.shared.b32 [%0], %1;" ::"r"(
                     convert_generic_ptr_to_smem_ptr(const_cast<int32*>(ptr))),
                 "r"(result)
                 : "memory");
#else
    asm volatile("st.relaxed.cta.shared.b32 [%0], %1;" ::"r"(
                     convert_generic_ptr_to_smem_ptr(const_cast<int32*>(ptr))),
                 "r"(result)
                 : "memory");
#endif
}


__device__ __forceinline__ int64 load_relaxed_shared(const int64* ptr)
{
    int64 result;
#if __CUDA_ARCH__ < 700
    asm volatile("ld.volatile.shared.b64 %0, [%1];"
                 : "=l"(result)
                 : "r"(convert_generic_ptr_to_smem_ptr(const_cast<int64*>(ptr)))
                 : "memory");
#else
    asm volatile("ld.relaxed.cta.shared.b64 %0, [%1];"
                 : "=l"(result)
                 : "r"(convert_generic_ptr_to_smem_ptr(const_cast<int64*>(ptr)))
                 : "memory");
#endif

    return result;
}


__device__ __forceinline__ void store_relaxed_shared(int64* ptr, int64 result)
{
#if __CUDA_ARCH__ < 700
    asm volatile("st.volatile.shared.b64 [%0], %1;" ::"r"(
                     convert_generic_ptr_to_smem_ptr(const_cast<int64*>(ptr))),
                 "l"(result)
                 : "memory");
#else
    asm volatile("st.relaxed.cta.shared.b64 [%0], %1;" ::"r"(
                     convert_generic_ptr_to_smem_ptr(const_cast<int64*>(ptr))),
                 "l"(result)
                 : "memory");
#endif
}


__device__ __forceinline__ float load_relaxed_shared(const float* ptr)
{
    float result;
#if __CUDA_ARCH__ < 700
    asm volatile("ld.volatile.shared.f32 %0, [%1];"
                 : "=f"(result)
                 : "r"(convert_generic_ptr_to_smem_ptr(const_cast<float*>(ptr)))
                 : "memory");
#else
    asm volatile("ld.relaxed.cta.shared.f32 %0, [%1];"
                 : "=f"(result)
                 : "r"(convert_generic_ptr_to_smem_ptr(const_cast<float*>(ptr)))
                 : "memory");
#endif

    return result;
}


__device__ __forceinline__ void store_relaxed_shared(float* ptr, float result)
{
#if __CUDA_ARCH__ < 700
    asm volatile("st.volatile.shared.f32 [%0], %1;" ::"r"(
                     convert_generic_ptr_to_smem_ptr(const_cast<float*>(ptr))),
                 "f"(result)
                 : "memory");
#else
    asm volatile("st.relaxed.cta.shared.f32 [%0], %1;" ::"r"(
                     convert_generic_ptr_to_smem_ptr(const_cast<float*>(ptr))),
                 "f"(result)
                 : "memory");
#endif
}


__device__ __forceinline__ double load_relaxed_shared(const double* ptr)
{
    double result;
#if __CUDA_ARCH__ < 700
    asm volatile(
        "ld.volatile.shared.f64 %0, [%1];"
        : "=d"(result)
        : "r"(convert_generic_ptr_to_smem_ptr(const_cast<double*>(ptr)))
        : "memory");
#else
    asm volatile(
        "ld.relaxed.cta.shared.f64 %0, [%1];"
        : "=d"(result)
        : "r"(convert_generic_ptr_to_smem_ptr(const_cast<double*>(ptr)))
        : "memory");
#endif

    return result;
}


__device__ __forceinline__ void store_relaxed_shared(double* ptr, double result)
{
#if __CUDA_ARCH__ < 700
    asm volatile("st.volatile.shared.f64 [%0], %1;" ::"r"(
                     convert_generic_ptr_to_smem_ptr(const_cast<double*>(ptr))),
                 "d"(result)
                 : "memory");
#else
    asm volatile("st.relaxed.cta.shared.f64 [%0], %1;" ::"r"(
                     convert_generic_ptr_to_smem_ptr(const_cast<double*>(ptr))),
                 "d"(result)
                 : "memory");
#endif
}


__device__ __forceinline__ int32 load_acquire_shared(const int32* ptr)
{
    int32 result;
#if __CUDA_ARCH__ < 700
    asm volatile("ld.volatile.shared.b32 %0, [%1];"
                 : "=r"(result)
                 : "r"(convert_generic_ptr_to_smem_ptr(const_cast<int32*>(ptr)))
                 : "memory");
#else
    asm volatile("ld.acquire.cta.shared.b32 %0, [%1];"
                 : "=r"(result)
                 : "r"(convert_generic_ptr_to_smem_ptr(const_cast<int32*>(ptr)))
                 : "memory");
#endif
    membar_acq_rel_shared();
    return result;
}


__device__ __forceinline__ void store_release_shared(int32* ptr, int32 result)
{
    membar_acq_rel_shared();
#if __CUDA_ARCH__ < 700
    asm volatile("st.volatile.shared.b32 [%0], %1;" ::"r"(
                     convert_generic_ptr_to_smem_ptr(const_cast<int32*>(ptr))),
                 "r"(result)
                 : "memory");
#else
    asm volatile("st.release.cta.shared.b32 [%0], %1;" ::"r"(
                     convert_generic_ptr_to_smem_ptr(const_cast<int32*>(ptr))),
                 "r"(result)
                 : "memory");
#endif
}


__device__ __forceinline__ int64 load_acquire_shared(const int64* ptr)
{
    int64 result;
#if __CUDA_ARCH__ < 700
    asm volatile("ld.volatile.shared.b64 %0, [%1];"
                 : "=l"(result)
                 : "r"(convert_generic_ptr_to_smem_ptr(const_cast<int64*>(ptr)))
                 : "memory");
#else
    asm volatile("ld.acquire.cta.shared.b64 %0, [%1];"
                 : "=l"(result)
                 : "r"(convert_generic_ptr_to_smem_ptr(const_cast<int64*>(ptr)))
                 : "memory");
#endif
    membar_acq_rel_shared();
    return result;
}


__device__ __forceinline__ void store_release_shared(int64* ptr, int64 result)
{
    membar_acq_rel_shared();
#if __CUDA_ARCH__ < 700
    asm volatile("st.volatile.shared.b64 [%0], %1;" ::"r"(
                     convert_generic_ptr_to_smem_ptr(const_cast<int64*>(ptr))),
                 "l"(result)
                 : "memory");
#else
    asm volatile("st.release.cta.shared.b64 [%0], %1;" ::"r"(
                     convert_generic_ptr_to_smem_ptr(const_cast<int64*>(ptr))),
                 "l"(result)
                 : "memory");
#endif
}


__device__ __forceinline__ float load_acquire_shared(const float* ptr)
{
    float result;
#if __CUDA_ARCH__ < 700
    asm volatile("ld.volatile.shared.f32 %0, [%1];"
                 : "=f"(result)
                 : "r"(convert_generic_ptr_to_smem_ptr(const_cast<float*>(ptr)))
                 : "memory");
#else
    asm volatile("ld.acquire.cta.shared.f32 %0, [%1];"
                 : "=f"(result)
                 : "r"(convert_generic_ptr_to_smem_ptr(const_cast<float*>(ptr)))
                 : "memory");
#endif
    membar_acq_rel_shared();
    return result;
}


__device__ __forceinline__ void store_release_shared(float* ptr, float result)
{
    membar_acq_rel_shared();
#if __CUDA_ARCH__ < 700
    asm volatile("st.volatile.shared.f32 [%0], %1;" ::"r"(
                     convert_generic_ptr_to_smem_ptr(const_cast<float*>(ptr))),
                 "f"(result)
                 : "memory");
#else
    asm volatile("st.release.cta.shared.f32 [%0], %1;" ::"r"(
                     convert_generic_ptr_to_smem_ptr(const_cast<float*>(ptr))),
                 "f"(result)
                 : "memory");
#endif
}


__device__ __forceinline__ double load_acquire_shared(const double* ptr)
{
    double result;
#if __CUDA_ARCH__ < 700
    asm volatile(
        "ld.volatile.shared.f64 %0, [%1];"
        : "=d"(result)
        : "r"(convert_generic_ptr_to_smem_ptr(const_cast<double*>(ptr)))
        : "memory");
#else
    asm volatile(
        "ld.acquire.cta.shared.f64 %0, [%1];"
        : "=d"(result)
        : "r"(convert_generic_ptr_to_smem_ptr(const_cast<double*>(ptr)))
        : "memory");
#endif
    membar_acq_rel_shared();
    return result;
}


__device__ __forceinline__ void store_release_shared(double* ptr, double result)
{
    membar_acq_rel_shared();
#if __CUDA_ARCH__ < 700
    asm volatile("st.volatile.shared.f64 [%0], %1;" ::"r"(
                     convert_generic_ptr_to_smem_ptr(const_cast<double*>(ptr))),
                 "d"(result)
                 : "memory");
#else
    asm volatile("st.release.cta.shared.f64 [%0], %1;" ::"r"(
                     convert_generic_ptr_to_smem_ptr(const_cast<double*>(ptr))),
                 "d"(result)
                 : "memory");
#endif
}


__device__ __forceinline__ int32 load_relaxed(const int32* ptr)
{
    int32 result;
#if __CUDA_ARCH__ < 700
    asm volatile("ld.volatile.b32 %0, [%1];"
                 : "=r"(result)
                 : "l"(const_cast<int32*>(ptr))
                 : "memory");
#else
    asm volatile("ld.relaxed.gpu.b32 %0, [%1];"
                 : "=r"(result)
                 : "l"(const_cast<int32*>(ptr))
                 : "memory");
#endif

    return result;
}


__device__ __forceinline__ void store_relaxed(int32* ptr, int32 result)
{
#if __CUDA_ARCH__ < 700
    asm volatile("st.volatile.b32 [%0], %1;" ::"l"(const_cast<int32*>(ptr)),
                 "r"(result)
                 : "memory");
#else
    asm volatile("st.relaxed.gpu.b32 [%0], %1;" ::"l"(const_cast<int32*>(ptr)),
                 "r"(result)
                 : "memory");
#endif
}


__device__ __forceinline__ int64 load_relaxed(const int64* ptr)
{
    int64 result;
#if __CUDA_ARCH__ < 700
    asm volatile("ld.volatile.b64 %0, [%1];"
                 : "=l"(result)
                 : "l"(const_cast<int64*>(ptr))
                 : "memory");
#else
    asm volatile("ld.relaxed.gpu.b64 %0, [%1];"
                 : "=l"(result)
                 : "l"(const_cast<int64*>(ptr))
                 : "memory");
#endif

    return result;
}


__device__ __forceinline__ void store_relaxed(int64* ptr, int64 result)
{
#if __CUDA_ARCH__ < 700
    asm volatile("st.volatile.b64 [%0], %1;" ::"l"(const_cast<int64*>(ptr)),
                 "l"(result)
                 : "memory");
#else
    asm volatile("st.relaxed.gpu.b64 [%0], %1;" ::"l"(const_cast<int64*>(ptr)),
                 "l"(result)
                 : "memory");
#endif
}


__device__ __forceinline__ float load_relaxed(const float* ptr)
{
    float result;
#if __CUDA_ARCH__ < 700
    asm volatile("ld.volatile.f32 %0, [%1];"
                 : "=f"(result)
                 : "l"(const_cast<float*>(ptr))
                 : "memory");
#else
    asm volatile("ld.relaxed.gpu.f32 %0, [%1];"
                 : "=f"(result)
                 : "l"(const_cast<float*>(ptr))
                 : "memory");
#endif

    return result;
}


__device__ __forceinline__ void store_relaxed(float* ptr, float result)
{
#if __CUDA_ARCH__ < 700
    asm volatile("st.volatile.f32 [%0], %1;" ::"l"(const_cast<float*>(ptr)),
                 "f"(result)
                 : "memory");
#else
    asm volatile("st.relaxed.gpu.f32 [%0], %1;" ::"l"(const_cast<float*>(ptr)),
                 "f"(result)
                 : "memory");
#endif
}


__device__ __forceinline__ double load_relaxed(const double* ptr)
{
    double result;
#if __CUDA_ARCH__ < 700
    asm volatile("ld.volatile.f64 %0, [%1];"
                 : "=d"(result)
                 : "l"(const_cast<double*>(ptr))
                 : "memory");
#else
    asm volatile("ld.relaxed.gpu.f64 %0, [%1];"
                 : "=d"(result)
                 : "l"(const_cast<double*>(ptr))
                 : "memory");
#endif

    return result;
}


__device__ __forceinline__ void store_relaxed(double* ptr, double result)
{
#if __CUDA_ARCH__ < 700
    asm volatile("st.volatile.f64 [%0], %1;" ::"l"(const_cast<double*>(ptr)),
                 "d"(result)
                 : "memory");
#else
    asm volatile("st.relaxed.gpu.f64 [%0], %1;" ::"l"(const_cast<double*>(ptr)),
                 "d"(result)
                 : "memory");
#endif
}


__device__ __forceinline__ int32 load_acquire(const int32* ptr)
{
    int32 result;
#if __CUDA_ARCH__ < 700
    asm volatile("ld.volatile.b32 %0, [%1];"
                 : "=r"(result)
                 : "l"(const_cast<int32*>(ptr))
                 : "memory");
#else
    asm volatile("ld.acquire.gpu.b32 %0, [%1];"
                 : "=r"(result)
                 : "l"(const_cast<int32*>(ptr))
                 : "memory");
#endif
    membar_acq_rel();
    return result;
}


__device__ __forceinline__ void store_release(int32* ptr, int32 result)
{
    membar_acq_rel();
#if __CUDA_ARCH__ < 700
    asm volatile("st.volatile.b32 [%0], %1;" ::"l"(const_cast<int32*>(ptr)),
                 "r"(result)
                 : "memory");
#else
    asm volatile("st.release.gpu.b32 [%0], %1;" ::"l"(const_cast<int32*>(ptr)),
                 "r"(result)
                 : "memory");
#endif
}


__device__ __forceinline__ int64 load_acquire(const int64* ptr)
{
    int64 result;
#if __CUDA_ARCH__ < 700
    asm volatile("ld.volatile.b64 %0, [%1];"
                 : "=l"(result)
                 : "l"(const_cast<int64*>(ptr))
                 : "memory");
#else
    asm volatile("ld.acquire.gpu.b64 %0, [%1];"
                 : "=l"(result)
                 : "l"(const_cast<int64*>(ptr))
                 : "memory");
#endif
    membar_acq_rel();
    return result;
}


__device__ __forceinline__ void store_release(int64* ptr, int64 result)
{
    membar_acq_rel();
#if __CUDA_ARCH__ < 700
    asm volatile("st.volatile.b64 [%0], %1;" ::"l"(const_cast<int64*>(ptr)),
                 "l"(result)
                 : "memory");
#else
    asm volatile("st.release.gpu.b64 [%0], %1;" ::"l"(const_cast<int64*>(ptr)),
                 "l"(result)
                 : "memory");
#endif
}


__device__ __forceinline__ float load_acquire(const float* ptr)
{
    float result;
#if __CUDA_ARCH__ < 700
    asm volatile("ld.volatile.f32 %0, [%1];"
                 : "=f"(result)
                 : "l"(const_cast<float*>(ptr))
                 : "memory");
#else
    asm volatile("ld.acquire.gpu.f32 %0, [%1];"
                 : "=f"(result)
                 : "l"(const_cast<float*>(ptr))
                 : "memory");
#endif
    membar_acq_rel();
    return result;
}


__device__ __forceinline__ void store_release(float* ptr, float result)
{
    membar_acq_rel();
#if __CUDA_ARCH__ < 700
    asm volatile("st.volatile.f32 [%0], %1;" ::"l"(const_cast<float*>(ptr)),
                 "f"(result)
                 : "memory");
#else
    asm volatile("st.release.gpu.f32 [%0], %1;" ::"l"(const_cast<float*>(ptr)),
                 "f"(result)
                 : "memory");
#endif
}


__device__ __forceinline__ double load_acquire(const double* ptr)
{
    double result;
#if __CUDA_ARCH__ < 700
    asm volatile("ld.volatile.f64 %0, [%1];"
                 : "=d"(result)
                 : "l"(const_cast<double*>(ptr))
                 : "memory");
#else
    asm volatile("ld.acquire.gpu.f64 %0, [%1];"
                 : "=d"(result)
                 : "l"(const_cast<double*>(ptr))
                 : "memory");
#endif
    membar_acq_rel();
    return result;
}


__device__ __forceinline__ void store_release(double* ptr, double result)
{
    membar_acq_rel();
#if __CUDA_ARCH__ < 700
    asm volatile("st.volatile.f64 [%0], %1;" ::"l"(const_cast<double*>(ptr)),
                 "d"(result)
                 : "memory");
#else
    asm volatile("st.release.gpu.f64 [%0], %1;" ::"l"(const_cast<double*>(ptr)),
                 "d"(result)
                 : "memory");
#endif
}


__device__ __forceinline__ thrust::complex<float> load_relaxed_shared(
    const thrust::complex<float>* ptr)
{
    float real_result;
    float imag_result;
#if __CUDA_ARCH__ < 700
    asm volatile("ld.volatile.shared.v2.f32 {%0, %1}, [%2];"
                 : "=f"(real_result), "=f"(imag_result)
                 : "r"(convert_generic_ptr_to_smem_ptr(
                     const_cast<thrust::complex<float>*>(ptr)))
                 : "memory");
#else
    asm volatile("ld.relaxed.cta.shared.v2.f32 {%0, %1}, [%2];"
                 : "=f"(real_result), "=f"(imag_result)
                 : "r"(convert_generic_ptr_to_smem_ptr(
                     const_cast<thrust::complex<float>*>(ptr)))
                 : "memory");
#endif
    return thrust::complex<float>{real_result, imag_result};
}


__device__ __forceinline__ void store_relaxed_shared(
    thrust::complex<float>* ptr, thrust::complex<float> result)
{
    auto real_result = result.real();
    auto imag_result = result.imag();
#if __CUDA_ARCH__ < 700
    asm volatile("st.volatile.shared.v2.f32 [%0], {%1, %2};" ::"r"(
                     convert_generic_ptr_to_smem_ptr(
                         const_cast<thrust::complex<float>*>(ptr))),
                 "f"(real_result), "f"(imag_result)
                 : "memory");
#else
    asm volatile("st.relaxed.cta.shared.v2.f32 [%0], {%1, %2};" ::"r"(
                     convert_generic_ptr_to_smem_ptr(
                         const_cast<thrust::complex<float>*>(ptr))),
                 "f"(real_result), "f"(imag_result)
                 : "memory");
#endif
}


__device__ __forceinline__ thrust::complex<double> load_relaxed_shared(
    const thrust::complex<double>* ptr)
{
    double real_result;
    double imag_result;
#if __CUDA_ARCH__ < 700
    asm volatile("ld.volatile.shared.v2.f64 {%0, %1}, [%2];"
                 : "=d"(real_result), "=d"(imag_result)
                 : "r"(convert_generic_ptr_to_smem_ptr(
                     const_cast<thrust::complex<double>*>(ptr)))
                 : "memory");
#else
    asm volatile("ld.relaxed.cta.shared.v2.f64 {%0, %1}, [%2];"
                 : "=d"(real_result), "=d"(imag_result)
                 : "r"(convert_generic_ptr_to_smem_ptr(
                     const_cast<thrust::complex<double>*>(ptr)))
                 : "memory");
#endif
    return thrust::complex<double>{real_result, imag_result};
}


__device__ __forceinline__ void store_relaxed_shared(
    thrust::complex<double>* ptr, thrust::complex<double> result)
{
    auto real_result = result.real();
    auto imag_result = result.imag();
#if __CUDA_ARCH__ < 700
    asm volatile("st.volatile.shared.v2.f64 [%0], {%1, %2};" ::"r"(
                     convert_generic_ptr_to_smem_ptr(
                         const_cast<thrust::complex<double>*>(ptr))),
                 "d"(real_result), "d"(imag_result)
                 : "memory");
#else
    asm volatile("st.relaxed.cta.shared.v2.f64 [%0], {%1, %2};" ::"r"(
                     convert_generic_ptr_to_smem_ptr(
                         const_cast<thrust::complex<double>*>(ptr))),
                 "d"(real_result), "d"(imag_result)
                 : "memory");
#endif
}


__device__ __forceinline__ thrust::complex<float> load_relaxed(
    const thrust::complex<float>* ptr)
{
    float real_result;
    float imag_result;
#if __CUDA_ARCH__ < 700
    asm volatile("ld.volatile.v2.f32 {%0, %1}, [%2];"
                 : "=f"(real_result), "=f"(imag_result)
                 : "l"(const_cast<thrust::complex<float>*>(ptr))
                 : "memory");
#else
    asm volatile("ld.relaxed.gpu.v2.f32 {%0, %1}, [%2];"
                 : "=f"(real_result), "=f"(imag_result)
                 : "l"(const_cast<thrust::complex<float>*>(ptr))
                 : "memory");
#endif
    return thrust::complex<float>{real_result, imag_result};
}


__device__ __forceinline__ void store_relaxed(thrust::complex<float>* ptr,
                                              thrust::complex<float> result)
{
    auto real_result = result.real();
    auto imag_result = result.imag();
#if __CUDA_ARCH__ < 700
    asm volatile("st.volatile.v2.f32 [%0], {%1, %2};" ::"l"(
                     const_cast<thrust::complex<float>*>(ptr)),
                 "f"(real_result), "f"(imag_result)
                 : "memory");
#else
    asm volatile("st.relaxed.gpu.v2.f32 [%0], {%1, %2};" ::"l"(
                     const_cast<thrust::complex<float>*>(ptr)),
                 "f"(real_result), "f"(imag_result)
                 : "memory");
#endif
}


__device__ __forceinline__ thrust::complex<double> load_relaxed(
    const thrust::complex<double>* ptr)
{
    double real_result;
    double imag_result;
#if __CUDA_ARCH__ < 700
    asm volatile("ld.volatile.v2.f64 {%0, %1}, [%2];"
                 : "=d"(real_result), "=d"(imag_result)
                 : "l"(const_cast<thrust::complex<double>*>(ptr))
                 : "memory");
#else
    asm volatile("ld.relaxed.gpu.v2.f64 {%0, %1}, [%2];"
                 : "=d"(real_result), "=d"(imag_result)
                 : "l"(const_cast<thrust::complex<double>*>(ptr))
                 : "memory");
#endif
    return thrust::complex<double>{real_result, imag_result};
}


__device__ __forceinline__ void store_relaxed(thrust::complex<double>* ptr,
                                              thrust::complex<double> result)
{
    auto real_result = result.real();
    auto imag_result = result.imag();
#if __CUDA_ARCH__ < 700
    asm volatile("st.volatile.v2.f64 [%0], {%1, %2};" ::"l"(
                     const_cast<thrust::complex<double>*>(ptr)),
                 "d"(real_result), "d"(imag_result)
                 : "memory");
#else
    asm volatile("st.relaxed.gpu.v2.f64 [%0], {%1, %2};" ::"l"(
                     const_cast<thrust::complex<double>*>(ptr)),
                 "d"(real_result), "d"(imag_result)
                 : "memory");
#endif
}


}  // namespace cuda
}  // namespace kernels
}  // namespace gko

#endif  // GKO_CUDA_COMPONENTS_MEMORY_CUH_
