// SPDX-FileCopyrightText: 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

template <typename ValueType>
__device__ __forceinline__ void simple_apply(
    const gko::batch::matrix::external::batch_item<const ValueType>& a,
    const ValueType* const __restrict__ b, ValueType* const __restrict__ c)
{
    a.simple_apply(a.batch_id,
                   {static_cast<size_type>(a.num_rows),
                    static_cast<size_type>(a.num_cols)},
                   b, c, a.payload);
}


template <typename ValueType>
__device__ __forceinline__ void advanced_apply(
    const ValueType alpha,
    const gko::batch::matrix::external::batch_item<const ValueType>& a,
    const ValueType* const __restrict__ b, const ValueType beta,
    ValueType* const __restrict__ c)
{
    a.advanced_apply(a.batch_id,
                     {static_cast<size_type>(a.num_rows),
                      static_cast<size_type>(a.num_cols)},
                     &alpha, b, &beta, c, a.payload);
}
