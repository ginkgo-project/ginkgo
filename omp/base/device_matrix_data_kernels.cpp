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

#include "core/base/device_matrix_data_kernels.hpp"


#include <algorithm>


#include <omp.h>


#include "core/base/allocator.hpp"


namespace gko {
namespace kernels {
namespace omp {
namespace components {


template <typename ValueType, typename IndexType>
void remove_zeros(std::shared_ptr<const DefaultExecutor> exec,
                  Array<ValueType>& values, Array<IndexType>& row_idxs,
                  Array<IndexType>& col_idxs)
{
    const auto size = values.get_num_elems();
    const auto num_threads = omp_get_max_threads();
    const auto per_thread = static_cast<size_type>(ceildiv(size, num_threads));
    vector<size_type> partial_counts(num_threads, {exec});
#pragma omp parallel num_threads(num_threads)
    {
        const auto tidx = static_cast<size_type>(omp_get_thread_num());
        const auto begin = per_thread * tidx;
        const auto end = std::min(size, begin + per_thread);
        for (auto i = begin; i < end; i++) {
            partial_counts[tidx] +=
                is_nonzero(values.get_const_data()[i]) ? 1 : 0;
        }
    }
    std::partial_sum(partial_counts.begin(), partial_counts.end(),
                     partial_counts.begin());
    auto nnz = static_cast<size_type>(partial_counts.back());
    if (nnz < size) {
        Array<ValueType> new_values{exec, nnz};
        Array<IndexType> new_row_idxs{exec, nnz};
        Array<IndexType> new_col_idxs{exec, nnz};
#pragma omp parallel num_threads(num_threads)
        {
            const auto tidx = static_cast<size_type>(omp_get_thread_num());
            const auto begin = per_thread * tidx;
            const auto end = std::min(size, begin + per_thread);
            auto out_idx = tidx == 0 ? size_type{} : partial_counts[tidx - 1];
            for (auto i = begin; i < end; i++) {
                auto val = values.get_const_data()[i];
                if (is_nonzero(val)) {
                    new_values.get_data()[out_idx] = val;
                    new_row_idxs.get_data()[out_idx] =
                        row_idxs.get_const_data()[i];
                    new_col_idxs.get_data()[out_idx] =
                        col_idxs.get_const_data()[i];
                    out_idx++;
                }
            }
        }
        values = std::move(new_values);
        row_idxs = std::move(new_row_idxs);
        col_idxs = std::move(new_col_idxs);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_REMOVE_ZEROS_KERNEL);


template <typename ValueType, typename IndexType>
void sort_row_major(std::shared_ptr<const DefaultExecutor> exec,
                    device_matrix_data<ValueType, IndexType>& data)
{
    Array<matrix_data_entry<ValueType, IndexType>> tmp{exec,
                                                       data.get_num_elems()};
    soa_to_aos(exec, data, tmp);
    std::sort(tmp.get_data(), tmp.get_data() + tmp.get_num_elems());
    aos_to_soa(exec, tmp, data);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_SORT_ROW_MAJOR_KERNEL);


}  // namespace components
}  // namespace omp
}  // namespace kernels
}  // namespace gko
