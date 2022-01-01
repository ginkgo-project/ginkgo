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

#include "core/matrix/sellp_kernels.hpp"


#include <ginkgo/core/base/math.hpp>


#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/base/kernel_launch_reduction.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Sellp matrix format namespace.
 *
 * @ingroup sellp
 */
namespace sellp {


void compute_slice_sets(std::shared_ptr<const DefaultExecutor> exec,
                        const Array<int64>& row_ptrs, size_type slice_size,
                        size_type stride_factor, size_type* slice_sets,
                        size_type* slice_lengths)
{
    const auto num_rows = row_ptrs.get_num_elems() - 1;
    const auto num_slices =
        static_cast<size_type>(ceildiv(num_rows, slice_size));
    run_kernel_row_reduction(
        exec,
        [] GKO_KERNEL(auto slice, auto local_row, auto row_ptrs,
                      auto slice_size, auto stride_factor, auto num_rows) {
            const auto row = slice * slice_size + local_row;
            return row < num_rows
                       ? static_cast<size_type>(
                             ceildiv(row_ptrs[row + 1] - row_ptrs[row],
                                     stride_factor) *
                             stride_factor)
                       : size_type{};
        },
        [] GKO_KERNEL(auto a, auto b) { return a > b ? a : b; },
        [] GKO_KERNEL(auto a) { return a; }, size_type{}, slice_lengths, 1,
        gko::dim<2>{num_slices, slice_size}, row_ptrs, slice_size,
        stride_factor, num_rows);
    exec->copy(num_slices, slice_lengths, slice_sets);
    components::prefix_sum(exec, slice_sets, num_slices + 1);
}


template <typename ValueType, typename IndexType>
void fill_in_matrix_data(
    std::shared_ptr<const DefaultExecutor> exec,
    const Array<matrix_data_entry<ValueType, IndexType>>& nonzeros,
    const int64* row_ptrs, matrix::Sellp<ValueType, IndexType>* output)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto nonzeros, auto row_ptrs, auto slice_size,
                      auto slice_sets, auto cols, auto values) {
            const auto row_begin = row_ptrs[row];
            const auto row_end = row_ptrs[row + 1];
            const auto slice = row / slice_size;
            const auto local_row = row % slice_size;
            const auto slice_begin = slice_sets[slice];
            const auto slice_end = slice_sets[slice + 1];
            const auto slice_length = slice_end - slice_begin;
            auto out_idx = slice_begin * slice_size + local_row;
            for (auto i = row_begin; i < row_begin + slice_length; i++) {
                cols[out_idx] = i < row_end ? nonzeros[i].column : 0;
                values[out_idx] = i < row_end ? unpack_member(nonzeros[i].value)
                                              : zero(values[out_idx]);
                out_idx += slice_size;
            }
        },
        output->get_size()[0], nonzeros, row_ptrs, output->get_slice_size(),
        output->get_const_slice_sets(), output->get_col_idxs(),
        output->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_FILL_IN_MATRIX_DATA_KERNEL);


}  // namespace sellp
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
