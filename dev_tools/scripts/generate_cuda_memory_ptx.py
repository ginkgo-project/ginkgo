#!/usr/bin/env python3
import dataclasses


@dataclasses.dataclass
class space:
    ptx_space_suffix: str
    ptx_scope_suffix: str
    fn_suffix: str
    ptr_expr: str
    ptr_constraint: str


@dataclasses.dataclass
class ordering:
    ptx_load_suffix: str
    fn_load_suffix: str
    ptx_store_suffix: str
    fn_store_suffix: str
    is_relaxed: bool


@dataclasses.dataclass
class type_desc:
    ptx_type_suffix: str
    val_constraint: str
    name: str


memory_spaces = [
    space(ptx_space_suffix=".shared", ptx_scope_suffix=".cta", fn_suffix="_shared",
          ptr_expr="convert_generic_ptr_to_smem_ptr({ptr})", ptr_constraint="r"),
    space(ptx_space_suffix="", ptx_scope_suffix=".gpu", fn_suffix="", ptr_expr="{ptr}", ptr_constraint="l")]
memory_orderings = [
    ordering(ptx_load_suffix=".relaxed", fn_load_suffix="_relaxed",
             ptx_store_suffix=".relaxed", fn_store_suffix="_relaxed", is_relaxed=True),
    ordering(ptx_load_suffix=".acquire", fn_load_suffix="_acquire",
             ptx_store_suffix=".release", fn_store_suffix="_release", is_relaxed=False)
]
types = [type_desc(ptx_type_suffix=".b32", val_constraint="r", name="int32"),
         type_desc(ptx_type_suffix=".b64", val_constraint="l", name="int64"),
         type_desc(ptx_type_suffix=".f32", val_constraint="f", name="float"),
         type_desc(ptx_type_suffix=".f64", val_constraint="d", name="double")]
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
""")

# relaxed
for s in memory_spaces:
    for o in memory_orderings:
        for t in types:
            membar_expression = "" if o.is_relaxed else f"membar_acq_rel{s.fn_suffix}();"
            const_ptr_expr = s.ptr_expr.format(
                ptr=f"const_cast<{t.name}*>(ptr)")
            mut_ptr_expr = s.ptr_expr.format(ptr="ptr")
            print(f"""
__device__ __forceinline__ {t.name} load{o.fn_load_suffix}{s.fn_suffix}(const {t.name}* ptr)
{{
    {t.name} result;
#if __CUDA_ARCH__ < 700
    asm volatile("ld.volatile{s.ptx_space_suffix}{t.ptx_type_suffix} %0, [%1];"
                 : "={t.val_constraint}"(result)
                 : "{s.ptr_constraint}"({const_ptr_expr})
                 : "memory");
#else
    asm volatile("ld{o.ptx_load_suffix}{s.ptx_scope_suffix}{s.ptx_space_suffix}{t.ptx_type_suffix} %0, [%1];"
                 : "={t.val_constraint}"(result)
                 : "{s.ptr_constraint}"({const_ptr_expr})
                 : "memory");
#endif
    {membar_expression}
    return result;
}}


__device__ __forceinline__ void store{o.fn_store_suffix}{s.fn_suffix}({t.name}* ptr, {t.name} result)
{{
    {membar_expression}
#if __CUDA_ARCH__ < 700
    asm volatile("st.volatile{s.ptx_space_suffix}{t.ptx_type_suffix} [%0], %1;"
                 :: "{s.ptr_constraint}"({mut_ptr_expr}), "{t.val_constraint}"(result)
                 : "memory");
#else
    asm volatile("st{o.ptx_store_suffix}{s.ptx_scope_suffix}{s.ptx_space_suffix}{t.ptx_type_suffix} [%0], %1;"
                 :: "{s.ptr_constraint}"({mut_ptr_expr}), "{t.val_constraint}"(result)
                 : "memory");
#endif
}}
""")

# vectorized relaxed loads for thrust::complex
types = [type_desc(ptx_type_suffix=".f32", val_constraint="f", name="float"),
         type_desc(ptx_type_suffix=".f64", val_constraint="d", name="double")]
for s in memory_spaces:
    for t in types:
        const_ptr_expr = s.ptr_expr.format(
            ptr=f"const_cast<thrust::complex<{t.name}>*>(ptr)")
        mut_ptr_expr = s.ptr_expr.format(ptr="ptr")
        print(f"""
__device__ __forceinline__ thrust::complex<{t.name}> load_relaxed{s.fn_suffix}(const thrust::complex<{t.name}>* ptr)
{{
    {t.name} real_result;
    {t.name} imag_result;
#if __CUDA_ARCH__ < 700
    asm volatile("ld.volatile{s.ptx_space_suffix}.v2{t.ptx_type_suffix} {{%0, %1}}, [%2];"
                 : "={t.val_constraint}"(real_result), "={t.val_constraint}"(imag_result)
                 : "{s.ptr_constraint}"({const_ptr_expr})
                 : "memory");
#else
    asm volatile("ld.relaxed{s.ptx_scope_suffix}{s.ptx_space_suffix}.v2{t.ptx_type_suffix} {{%0, %1}}, [%2];"
                 : "={t.val_constraint}"(real_result), "={t.val_constraint}"(imag_result)
                 : "{s.ptr_constraint}"({const_ptr_expr})
                 : "memory");
#endif
    return thrust::complex<{t.name}>{{real_result, imag_result}};
}}


__device__ __forceinline__ void store_relaxed{s.fn_suffix}(thrust::complex<{t.name}>* ptr, thrust::complex<{t.name}> result)
{{
    auto real_result = result.real();
    auto imag_result = result.imag();
#if __CUDA_ARCH__ < 700
    asm volatile("st.volatile{s.ptx_space_suffix}.v2{t.ptx_type_suffix} [%0], {{%1, %2}};"
                 :: "{s.ptr_constraint}"({mut_ptr_expr}), "{t.val_constraint}"(real_result), "{t.val_constraint}"(imag_result)
                 : "memory");
#else
    asm volatile("st.relaxed{s.ptx_scope_suffix}{s.ptx_space_suffix}.v2{t.ptx_type_suffix} [%0], {{%1, %2}};"
                 :: "{s.ptr_constraint}"({mut_ptr_expr}), "{t.val_constraint}"(real_result), "{t.val_constraint}"(imag_result)
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
