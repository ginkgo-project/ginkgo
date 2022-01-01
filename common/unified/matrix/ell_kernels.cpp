/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include "core/matrix/ell_kernels.hpp"


#include <ginkgo/core/base/math.hpp>


#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/base/kernel_launch_reduction.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Ell matrix format namespace.
 *
 * @ingroup ell
 */
namespace ell {


template <typename IndexType>
void compute_max_row_nnz(std::shared_ptr<const DefaultExecutor> exec,
                         const Array<IndexType>& row_ptrs, size_type& max_nnz)
{
    Array<size_type> result{exec, 1};
    run_kernel_reduction(
        exec,
        [] GKO_KERNEL(auto i, auto row_ptrs) {
            return row_ptrs[i + 1] - row_ptrs[i];
        },
        [] GKO_KERNEL(auto a, auto b) { return a > b ? a : b; },
        [] GKO_KERNEL(auto a) { return a; }, size_type{}, result.get_data(),
        row_ptrs.get_num_elems() - 1, row_ptrs);
    max_nnz = exec->copy_val_to_host(result.get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_ELL_COMPUTE_MAX_ROW_NNZ_KERNEL);


template <typename ValueType, typename IndexType>
void fill_in_matrix_data(
    std::shared_ptr<const DefaultExecutor> exec,
    const Array<matrix_data_entry<ValueType, IndexType>>& nonzeros,
    const int64* row_ptrs, matrix::Ell<ValueType, IndexType>* output)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto nonzeros, auto row_ptrs, auto stride,
                      auto num_cols, auto cols, auto values) {
            const auto begin = row_ptrs[row];
            const auto end = row_ptrs[row + 1];
            auto out_idx = row;
            for (auto i = begin; i < begin + num_cols; i++) {
                cols[out_idx] = i < end ? nonzeros[i].column : 0;
                values[out_idx] = i < end ? unpack_member(nonzeros[i].value)
                                          : zero(values[out_idx]);
                out_idx += stride;
            }
        },
        output->get_size()[0], nonzeros, row_ptrs, output->get_stride(),
        output->get_num_stored_elements_per_row(), output->get_col_idxs(),
        output->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_FILL_IN_MATRIX_DATA_KERNEL);


}  // namespace ell
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
