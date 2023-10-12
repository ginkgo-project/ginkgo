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

template <typename ValueType, typename IndexType>
__dpct_inline__ void simple_apply_kernel(
    const gko::batch::matrix::ell::batch_item<const ValueType, IndexType>& mat,
    const gko::batch::multi_vector::batch_item<const ValueType>& b,
    const gko::batch::multi_vector::batch_item<ValueType>& x,
    sycl::nd_item<3>& item_ct1)
{
    for (int tidx = item_ct1.get_local_linear_id(); tidx < mat.num_rows;
         tidx += item_ct1.get_local_range().size()) {
        auto temp = zero<ValueType>();
        for (size_type idx = 0; idx < mat.num_stored_elems_per_row; idx++) {
            const auto col_idx = mat.col_idxs[tidx + idx * mat.stride];
            if (col_idx == invalid_index<IndexType>()) {
                break;
            } else {
                temp += mat.values[tidx + idx * mat.stride] *
                        b.values[col_idx * b.stride];
            }
        }
        x.values[tidx * x.stride] = temp;
    }
}


template <typename ValueType, typename IndexType>
__dpct_inline__ void advanced_apply_kernel(
    const gko::batch::multi_vector::batch_item<const ValueType>& alpha,
    const gko::batch::matrix::ell::batch_item<const ValueType, IndexType>& mat,
    const gko::batch::multi_vector::batch_item<const ValueType>& b,
    const gko::batch::multi_vector::batch_item<const ValueType>& beta,
    const gko::batch::multi_vector::batch_item<ValueType>& x,
    sycl::nd_item<3>& item_ct1)
{
    for (int tidx = item_ct1.get_local_linear_id(); tidx < mat.num_rows;
         tidx += item_ct1.get_local_range().size()) {
        auto temp = zero<ValueType>();
        for (size_type idx = 0; idx < mat.num_stored_elems_per_row; idx++) {
            const auto col_idx = mat.col_idxs[tidx + idx * mat.stride];
            if (col_idx == invalid_index<IndexType>()) {
                break;
            } else {
                temp += mat.values[tidx + idx * mat.stride] *
                        b.values[col_idx * b.stride];
            }
        }
        x.values[tidx * x.stride] =
            alpha.values[0] * temp + beta.values[0] * x.values[tidx * x.stride];
    }
}
