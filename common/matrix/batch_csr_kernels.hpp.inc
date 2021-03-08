/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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


namespace {

// template <typename ValueType, typename IndexType>
// __device__ void device_batch_apply(typename gko::matrix::BatchCsr<ValueType,
// IndexType>::device_obj batch_csr, ValueType a, IndexType b)
// {
//     printf("%lf %ld\n", batch_csr.value, batch_csr.index);
// }


// template <typename ValueType, typename IndexType>
// __global__ void batch_apply_kernel(typename gko::matrix::BatchCsr<ValueType,
// IndexType>::device_obj batch_csr, ValueType a, IndexType b)
// {
//     device_batch_apply(batch_csr, a, b);
// }

// __global__ void batch_apply_kernel(typename gko::matrix::BatchCsr<float,
// int>::device_obj batch_csr)
// {
//     device_batch_apply(batch_csr);
// }

template <typename ValueType, typename IndexType>
struct csr_struct {
    ValueType value;
    IndexType index;
};

template <typename ValueType, typename IndexType>
csr_struct<cuda_type<ValueType>, IndexType> prepare(
    const matrix::BatchCsr<ValueType, IndexType> *a)
{
    /* get the information */
    csr_struct<cuda_type<ValueType>, IndexType> csr;
    csr.value = 1.0;
    csr.index = 0.0;
    return csr;
}


template <typename ValueType, typename IndexType>
__device__ void device_batch_apply(csr_struct<ValueType, IndexType> batch_csr)
{
    auto a = batch_csr.value;
    double d = real(a);
    printf("CSR_APPLY %lf\n", d);
}


template <typename ValueType, typename IndexType>
__global__ void batch_apply_kernel(csr_struct<ValueType, IndexType> batch_csr,
                                   dense_struct<ValueType> batch_dense)
{
    // device_batch_apply(batch_csr, a, b);
    // printf("%lf\n", batch_csr.value);
    device_batch_apply(batch_csr);
    device_batch_apply(batch_dense);
}

}  // namespace
