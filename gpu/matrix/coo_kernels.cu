/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/matrix/coo_kernels.hpp"


#include "core/base/exception_helpers.hpp"

#include "gpu/base/cusparse_bindings.hpp"
#include "gpu/base/types.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "core/base/math.hpp"


namespace gko {
namespace kernels {
namespace gpu {
namespace coo {

namespace {

__forceinline__ __device__ static float atomic_add(float *addr, float val)
{
    return atomicAdd(addr, val);
}

#if (defined(CUDA_VERSION) && (CUDA_VERSION < 8000)) || \
    (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600))
__forceinline__ __device__ static double atomic_add(double *addr, double val)
{
    double old = *addr, assumed;
    do {
        assumed = old;
        old = __longlong_as_double(atomicCAS(
            (unsigned long long int *)addr, __double_as_longlong(assumed),
            __double_as_longlong(val + assumed)));
    } while (assumed != old);

    return old;
}
#else
__forceinline__ __device__ static double atomic_add(double *addr, double val)
{
    return atomicAdd(addr, val);
}
#endif


__forceinline__ __device__ static cuDoubleComplex atomic_add(cuDoubleComplex *address,
                                                     cuDoubleComplex val)
{
    // Seperate to real part and imag part
    // real part
    atomic_add(&(address->x), val.x);
    // imag part
    atomic_add(&(address->y), val.y);
    return *address;
}

__forceinline__ __device__ static cuComplex atomic_add(cuComplex *address,
                                               cuComplex val)
{
    // Seperate to real part and imag part
    // real part
    atomic_add(&(address->x), val.x);
    // imag part
    atomic_add(&(address->y), val.y);
    return *address;
}

__forceinline__ __device__ static thrust::complex<float> atomic_add(thrust::complex<float> *address,
                                            thrust::complex<float> val)
{
    cuComplex *cuaddr = reinterpret_cast<cuComplex *>(address);
    cuComplex *cuval = reinterpret_cast<cuComplex *>(&val);
    atomic_add(cuaddr, *cuval);
    return *address;
}

__forceinline__ __device__ static thrust::complex<double> atomic_add(thrust::complex<double> *address,
                                             thrust::complex<double> val)
{
    cuDoubleComplex *cuaddr = reinterpret_cast<cuDoubleComplex *>(address);
    cuDoubleComplex *cuval = reinterpret_cast<cuDoubleComplex *>(&val);
    atomic_add(cuaddr, *cuval);
    return *address;
}


}  // namespace

template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const GpuExecutor> exec,
          const matrix::Coo<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b,
          matrix::Dense<ValueType> *c) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const GpuExecutor> exec,
                   const matrix::Dense<ValueType> *alpha,
                   const matrix::Coo<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_ADVANCED_SPMV_KERNEL);


template <typename IndexType>
void convert_row_idxs_to_ptrs(std::shared_ptr<const GpuExecutor> exec,
                              const IndexType *idxs, size_type num_nonzeros,
                              IndexType *ptrs,
                              size_type length) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_COO_CONVERT_ROW_IDXS_TO_PTRS_KERNEL);


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const GpuExecutor> exec,
               matrix::Coo<ValueType, IndexType> *trans,
               const matrix::Coo<ValueType, IndexType> *orig) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const GpuExecutor> exec,
                    matrix::Coo<ValueType, IndexType> *trans,
                    const matrix::Coo<ValueType, IndexType> *orig)
    NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(
    std::shared_ptr<const GpuExecutor> exec, matrix::Dense<ValueType> *result,
    const matrix::Coo<ValueType, IndexType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_CONVERT_TO_DENSE_KERNEL);


}  // namespace coo
}  // namespace gpu
}  // namespace kernels
}  // namespace gko
