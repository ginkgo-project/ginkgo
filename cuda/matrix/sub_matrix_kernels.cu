/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include "core/matrix/sub_matrix_kernels.hpp"


#include <algorithm>
#include <iterator>
#include <numeric>
#include <utility>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/allocator.hpp"
#include "core/base/iterator_factory.hpp"
#include "core/components/prefix_sum.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The Compressed sparse row matrix format namespace.
 * @ref Sub_Matrix
 * @ingroup sub_matrix
 */
namespace sub_matrix {


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const DefaultExecutor> exec,
          const matrix::SubMatrix<matrix::Csr<ValueType, IndexType>> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    // auto a = sub_mat->get_sub_matrix();
    // auto overlaps = sub_mat->get_overlap_mtxs();
    // auto num_overlaps = overlaps.size();
    // auto row_offset = sub_mat->get_overlap_sizes().get_data();
    // auto overlap_sizes = std::vector<int>(num_overlaps, 0);
    // auto left_ov_bound = sub_mat->get_left_overlap_bound();
    // bool fl = true;
    // for (int i = 1; i < num_overlaps; ++i) {
    //     overlap_sizes[i] = overlap_sizes[i - 1] + overlaps[i]->get_size()[1];
    //     if (i > left_ov_bound && fl) {
    //         overlap_sizes[i] += a->get_size()[1];
    //         fl = false;
    //     }
    // }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SUB_MATRIX_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Dense<ValueType> *alpha,
    const matrix::SubMatrix<matrix::Csr<ValueType, IndexType>> *a,
    const matrix::Dense<ValueType> *b, const matrix::Dense<ValueType> *beta,
    matrix::Dense<ValueType> *c)
{}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SUB_MATRIX_ADVANCED_SPMV_KERNEL);


}  // namespace sub_matrix
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
