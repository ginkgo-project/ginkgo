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

#include "core/matrix/hybrid_kernels.hpp"


#include <omp.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/format_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/ell_kernels.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The Hybrid matrix format namespace.
 *
 * @ingroup hybrid
 */
namespace hybrid {


template <typename ValueType, typename IndexType>
void split_matrix_data(
    std::shared_ptr<const DefaultExecutor> exec,
    const Array<matrix_data_entry<ValueType, IndexType>>& data,
    const int64* row_ptrs, size_type ell_limit, size_type num_rows,
    Array<matrix_data_entry<ValueType, IndexType>>& ell_data,
    Array<matrix_data_entry<ValueType, IndexType>>& coo_data)
{
    auto data_ptr = data.get_const_data();
    size_type ell_nnz{};
    for (size_type row = 0; row < num_rows; row++) {
        ell_nnz +=
            std::min<size_type>(ell_limit, row_ptrs[row + 1] - row_ptrs[row]);
    }
    ell_data.resize_and_reset(ell_nnz);
    coo_data.resize_and_reset(data.get_num_elems() - ell_nnz);
    size_type ell_nz{};
    size_type coo_nz{};
    for (size_type row = 0; row < num_rows; row++) {
        size_type local_ell_nnz{};
        for (auto i = row_ptrs[row]; i < row_ptrs[row + 1]; i++) {
            if (local_ell_nnz < ell_limit) {
                ell_data.get_data()[ell_nz] = data.get_const_data()[i];
                ell_nz++;
                local_ell_nnz++;
            } else {
                coo_data.get_data()[coo_nz] = data.get_const_data()[i];
                coo_nz++;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_HYBRID_SPLIT_MATRIX_DATA_KERNEL);


}  // namespace hybrid
}  // namespace omp
}  // namespace kernels
}  // namespace gko
