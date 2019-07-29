/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include "core/solver/trs_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>


#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The TRS solver namespace.
 *
 * @ingroup trs
 */
namespace trs {


constexpr int default_block_size = 512;

template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const CudaExecutor> exec,
              const matrix::Csr<ValueType, IndexType> *matrix)
    GKO_NOT_IMPLEMENTED;

template <typename ValueType, typename IndexType>
void solve(std::shared_ptr<const CudaExecutor> exec,
           const matrix::Csr<ValueType, IndexType> *matrix,
           const matrix::Dense<ValueType> *b,
           matrix::Dense<ValueType> *x) GKO_NOT_IMPLEMENTED;
// {
//     if (cusparse::is_supported<ValueType, IndexType>::value) {
//         // TODO: add implementation for int64 and multiple RHS
//         auto handle = exec->get_cusparse_handle();
//         auto descr = cusparse::create_mat_descr();
//         GKO_ASSERT_NO_CUSPARSE_ERRORS(
//             cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

//         auto row_ptrs = matrix->get_const_row_ptrs();
//         auto col_idxs = matrix->get_const_col_idxs();
//         auto values = matrix->get_const_values();
//         auto alpha = one<ValueType>();
//         auto beta = zero<ValueType>();
//         if (b->get_stride() != 1 || x->get_stride() != 1)
//         GKO_NOT_IMPLEMENTED;

//         cusparse::spmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//                        matrix->get_size()[0], matrix->get_size()[1],
//                        matrix->get_num_stored_elements(), &alpha, descr,
//                        values, row_ptrs, col_idxs, b->get_const_values(),
//                        &beta, x->get_values());

//         GKO_ASSERT_NO_CUSPARSE_ERRORS(
//             cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE));

//         cusparse::destroy(descr);
//     } else {
//         GKO_NOT_IMPLEMENTED;
//     }
// }

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_TRS_GENERATE_KERNEL);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_TRS_SOLVE_KERNEL);


}  // namespace trs
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
