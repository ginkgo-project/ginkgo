// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TRICK_REDUCTION_HPP_
#define TRICK_REDUCTION_HPP_


#include <cuda/components/reduction.cuh>


namespace gko {
namespace kernels {
namespace cuda {


template <typename Operator, typename ValueType>
void __device__ reduce_array_t(size_type size, const ValueType *__restrict__ source, ValueType *__restrict__ result, Operator reduce_op = Operator{})
{
    auto tid = threadIdx.x;
    reduce_array(size, source, result, reduce_op);
}


}  // namespace cuda
}  // namespace kernels
}  // namespace gko

#endif  // TRICK_REDUCTION_HPP_
