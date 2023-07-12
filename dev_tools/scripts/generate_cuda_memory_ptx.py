#!/usr/bin/env python3
import os
memory_spaces = [(".shared", ".cta", "_shared", "convert_generic_ptr_to_smem_ptr(ptr)", "r"), ("", ".gpu", "", "ptr", "l")]
memory_orderings = [
    (".relaxed", "_relaxed", ".relaxed", "_relaxed", True),
    (".acquire", "_acquire", ".release", "_release", False)
    ]
sizes=[(".b32", "r", "int32", 4), (".b64", "l", "int64", 8), (".f32", "f", "float", 4), (".f64", "d", "double", 8)]
# header
print("""/*******************************<GINKGO LICENSE>******************************
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
""")

# relaxed
for memory_space_suffix, scope_suffix, function_memory_space_suffix, ptr_name, ptr_constraint in memory_spaces:
    for volta_load_ordering_suffix, load_function_ordering_suffix, volta_store_ordering_suffix, store_function_ordering_suffix, is_relaxed in memory_orderings:
        for size_suffix, constraint, typename, size in sizes:
            membar_expression = "" if is_relaxed else f"membar_acq_rel{function_memory_space_suffix}();"
            print(f"""
__device__ __forceinline__ {typename} load{load_function_ordering_suffix}{function_memory_space_suffix}({typename}* ptr)
{{
    {typename} result;
#if __CUDA_ARCH__ < 700
    asm volatile("ld.volatile{memory_space_suffix}{size_suffix} %0, [%1];"
                 : "={constraint}"(result)
                 : "{ptr_constraint}"({ptr_name})
                 : "memory");
#else
    asm volatile("ld{volta_load_ordering_suffix}{scope_suffix}{memory_space_suffix}{size_suffix} %0, [%1];"
                 : "={constraint}"(result)
                 : "{ptr_constraint}"({ptr_name})
                 : "memory");
#endif
    {membar_expression}
    return result;
}}


__device__ __forceinline__ void store{store_function_ordering_suffix}{function_memory_space_suffix}({typename}* ptr, {typename} result)
{{
    {membar_expression}
#if __CUDA_ARCH__ < 700
    asm volatile("st.volatile{memory_space_suffix}{size_suffix} [%0], %1;"
                 :: "{ptr_constraint}"({ptr_name}), "{constraint}"(result)
                 : "memory");
#else
    asm volatile("st{volta_store_ordering_suffix}{scope_suffix}{memory_space_suffix}{size_suffix} [%0], %1;"
                 :: "{ptr_constraint}"({ptr_name}), "{constraint}"(result)
                 : "memory");
#endif
}}
""")

# vectorized relaxed loads for thrust::complex
sizes=[(".f32", "f", "float", 4), (".f64", "d", "double", 8)]
for memory_space_suffix, scope_suffix, function_memory_space_suffix, ptr_name, ptr_constraint in memory_spaces:
    for size_suffix, constraint, typename, size in sizes:
        print(f"""
__device__ __forceinline__ thrust::complex<{typename}> load_relaxed{function_memory_space_suffix}(thrust::complex<{typename}>* ptr)
{{
    {typename} real_result;
    {typename} imag_result;
#if __CUDA_ARCH__ < 700
    asm volatile("ld.volatile{memory_space_suffix}.v2{size_suffix} {{%0, %1}}, [%2];"
                 : "={constraint}"(real_result), "={constraint}"(imag_result)
                 : "{ptr_constraint}"({ptr_name})
                 : "memory");
#else
    asm volatile("ld.relaxed{scope_suffix}{memory_space_suffix}.v2{size_suffix} {{%0, %1}}, [%2];"
                 : "={constraint}"(real_result), "={constraint}"(imag_result)
                 : "{ptr_constraint}"({ptr_name})
                 : "memory");
#endif
    return thrust::complex<{typename}>{{real_result, imag_result}};
}}


__device__ __forceinline__ void store_relaxed{function_memory_space_suffix}(thrust::complex<{typename}>* ptr, thrust::complex<{typename}> result)
{{
    auto real_result = result.real();
    auto imag_result = result.imag();
#if __CUDA_ARCH__ < 700
    asm volatile("st.volatile{memory_space_suffix}.v2{size_suffix} [%0], {{%1, %2}};"
                 :: "{ptr_constraint}"({ptr_name}), "{constraint}"(real_result), "{constraint}"(imag_result)
                 : "memory");
#else
    asm volatile("st.relaxed{scope_suffix}{memory_space_suffix}.v2{size_suffix} [%0], {{%1, %2}};"
                 :: "{ptr_constraint}"({ptr_name}), "{constraint}"(real_result), "{constraint}"(imag_result)
                 : "memory");
#endif
}}
""")

print("""
}  // namespace cuda
}  // namespace kernels
}  // namespace gko

#endif  // GKO_CUDA_COMPONENTS_MEMORY_CUH_
""")