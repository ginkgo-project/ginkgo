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

#include "core/solver/upper_trs_kernels.hpp"


#include <memory>


#include <cuda.h>
#include <cusparse.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/solver/upper_trs.hpp>


#include "core/base/mixed_precision_types.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/solver/common_trs_kernels.cuh"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The UPPER_TRS solver namespace.
 *
 * @ingroup upper_trs
 */
namespace upper_trs {


void should_perform_transpose(std::shared_ptr<const CudaExecutor> exec,
                              bool& do_transpose)
{
    should_perform_transpose_kernel(exec, do_transpose);
}


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const CudaExecutor> exec,
              const matrix::Csr<ValueType, IndexType>* matrix,
              std::shared_ptr<solver::SolveStruct>& solve_struct,
              const gko::size_type num_rhs)
{
    if (matrix->get_strategy()->get_name() == "sparselib") {
        generate_kernel<ValueType, IndexType>(exec, matrix, solve_struct,
                                              num_rhs, true);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_UPPER_TRS_GENERATE_KERNEL);


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void solve(std::shared_ptr<const CudaExecutor> exec,
           const matrix::Csr<MatrixValueType, IndexType>* matrix,
           const solver::SolveStruct* solve_struct,
           matrix::Dense<InputValueType>* trans_b,
           matrix::Dense<OutputValueType>* trans_x,
           const matrix::Dense<InputValueType>* b,
           matrix::Dense<OutputValueType>* x)
{
    bool all_same_value_type =
        std::is_same<MatrixValueType, InputValueType>::value &&
        std::is_same<MatrixValueType, OutputValueType>::value;
    if (matrix->get_strategy()->get_name() == "sparselib" &&
        all_same_value_type) {
        solve_kernel<MatrixValueType, IndexType>(
            exec, matrix, solve_struct,
            gko::as<matrix::Dense<MatrixValueType>>(trans_b),
            gko::as<matrix::Dense<MatrixValueType>>(trans_x),
            gko::as<matrix::Dense<MatrixValueType>>(b),
            gko::as<matrix::Dense<MatrixValueType>>(x));
    } else {
        sptrsv_naive_caching<true>(exec, matrix, b, x);
    }
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_UPPER_TRS_SOLVE_KERNEL);


}  // namespace upper_trs
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
