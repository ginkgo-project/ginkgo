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
#ifndef GKO_DPCPP_MATRIX_BATCH_DIAGONAL_KERNELS_HPP_
#define GKO_DPCPP_MATRIX_BATCH_DIAGONAL_KERNELS_HPP_

#include "core/matrix/batch_diagonal_kernels.hpp"


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>


#include "core/matrix/batch_struct.hpp"


#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The BatchDense matrix format namespace.
 * @ref BatchDense
 * @ingroup batch_diagonal
 */
namespace batch_diagonal {


template <typename ValueType>
inline void apply_kernel(const int nrows, const int ncols,
                         const ValueType* const diag, const int nrhs,
                         const size_type b_stride, const ValueType* const b,
                         const size_type x_stride, ValueType* const x,
                         sycl::nd_item<3>& item_ct1)
{
    const int local_id = item_ct1.get_local_linear_id();
    const int local_range = item_ct1.get_local_range().size();

    const int mindim = min(nrows, ncols);
    for (int iz = local_id; iz < mindim * nrhs; iz += local_range) {
        const int row = iz / nrhs;
        const int col = iz % nrhs;
        x[row * x_stride + col] = diag[row] * b[row * b_stride + col];
    }
    for (int iz = local_id + mindim * nrhs; iz < nrows * nrhs;
         iz += local_range) {
        const int row = iz / nrhs;
        const int col = iz % nrhs;
        x[row * x_stride + col] = zero<ValueType>();
    }
}


template <typename ValueType>
inline void apply_in_place_kernel(const int num_rows, const size_type stride,
                                  const int num_rhs,
                                  const ValueType* const diag_vec,
                                  ValueType* const a,
                                  sycl::nd_item<3>& item_ct1)
{
    for (int iz = item_ct1.get_local_linear_id(); iz < num_rows * num_rhs;
         iz += item_ct1.get_local_range().size()) {
        const int row = iz / num_rhs;
        const int col = iz % num_rhs;
        a[row * stride + col] *= diag_vec[row];
    }
}


}  // namespace batch_diagonal
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif
