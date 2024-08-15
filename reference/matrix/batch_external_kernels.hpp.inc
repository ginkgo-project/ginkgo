// SPDX-FileCopyrightText: 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

template <typename ValueType>
void simple_apply_kernel(
    const gko::batch::matrix::external::batch_item<const ValueType>& a,
    const gko::batch::multi_vector::batch_item<const ValueType>& b,
    const gko::batch::multi_vector::batch_item<ValueType>& c)
{
    a.simple_apply(a.batch_id,
                   {static_cast<size_type>(a.num_rows),
                    static_cast<size_type>(a.num_cols)},
                   b.values, c.values, a.payload);
}


template <typename ValueType>
void advanced_apply_kernel(
    const ValueType alpha,
    const gko::batch::matrix::external::batch_item<const ValueType>& a,
    const gko::batch::multi_vector::batch_item<const ValueType>& b,
    const ValueType beta,
    const gko::batch::multi_vector::batch_item<ValueType>& c)
{
    a.advanced_apply(a.batch_id,
                     {static_cast<size_type>(a.num_rows),
                      static_cast<size_type>(a.num_cols)},
                     &alpha, b.values, &beta, c.values, a.payload);
}
