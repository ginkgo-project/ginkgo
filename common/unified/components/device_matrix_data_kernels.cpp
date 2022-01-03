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

#include "core/components/device_matrix_data_kernels.hpp"


#include <ginkgo/core/base/types.hpp>


#include "common/unified/base/kernel_launch.hpp"
#include "core/components/fill_array_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace components {


template <typename ValueType, typename IndexType, typename RowPtrType>
void build_row_ptrs(std::shared_ptr<const DefaultExecutor> exec,
                    const Array<matrix_data_entry<ValueType, IndexType>>& data,
                    size_type num_rows, RowPtrType* row_ptrs)
{
    if (data.get_num_elems() == 0) {
        fill_array(exec, row_ptrs, num_rows + 1, RowPtrType{});
    } else {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto i, auto num_nonzeros, auto num_rows,
                          auto nonzeros, auto row_ptrs) {
                auto begin_row = i == 0 ? size_type{} : nonzeros[i - 1].row;
                auto end_row = i == num_nonzeros ? num_rows : nonzeros[i].row;
                for (auto row = begin_row; row < end_row; row++) {
                    row_ptrs[row + 1] = i;
                }
                if (i == 0) {
                    row_ptrs[0] = 0;
                }
            },
            data.get_num_elems() + 1, data.get_num_elems(), num_rows, data,
            row_ptrs);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_BUILD_ROW_PTRS_KERNEL32);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_BUILD_ROW_PTRS_KERNEL64);


template <typename IndexType, typename RowPtrType>
void build_row_ptrs_from_idxs(std::shared_ptr<const DefaultExecutor> exec,
                              const Array<IndexType>& row_idxs,
                              size_type num_rows, RowPtrType* row_ptrs)
{
    if (row_idxs.get_num_elems() == 0) {
        fill_array(exec, row_ptrs, num_rows + 1, RowPtrType{});
    } else {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto i, auto num_idxs, auto num_rows, auto row_idxs,
                          auto row_ptrs) {
                auto begin_row = i == 0 ? size_type{} : row_idxs[i - 1];
                auto end_row = i == num_idxs ? num_rows : row_idxs[i];
                for (auto row = begin_row; row < end_row; row++) {
                    row_ptrs[row + 1] = i;
                }
                if (i == 0) {
                    row_ptrs[0] = 0;
                }
            },
            row_idxs.get_num_elems() + 1, row_idxs.get_num_elems(), num_rows,
            row_idxs, row_ptrs);
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_BUILD_ROW_PTRS_FROM_IDXS_KERNEL32);
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_BUILD_ROW_PTRS_FROM_IDXS_KERNEL64);


}  // namespace components
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
