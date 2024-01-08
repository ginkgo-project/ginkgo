// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TRICK_SORTING_HPP_
#define TRICK_SORTING_HPP_


#include <type_traits>


#include "cuda/components/sorting.cuh"


namespace gko {
namespace kernels {
namespace cuda {


template <int num_elements, int num_local, typename ValueType>
__forceinline__ __device__ void bitonic_sort_t(ValueType *local_elements, ValueType *shared_elements)
{
    auto tidx = threadIdx.x;
    bitonic_sort<num_elements, num_local>(local_elements, shared_elements);
}


}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // TRICK_SORTING_HPP_
