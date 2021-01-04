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

#include "core/solver/upper_trs_kernels.hpp"


#include <memory>


#include <hip/hip_runtime.h>
#include <hipsparse.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/solver/upper_trs.hpp>


#include "hip/base/hipsparse_bindings.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/solver/common_trs_kernels.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The UPPER_TRS solver namespace.
 *
 * @ingroup upper_trs
 */
namespace upper_trs {


void should_perform_transpose(std::shared_ptr<const HipExecutor> exec,
                              bool &do_transpose)
{
    should_perform_transpose_kernel(exec, do_transpose);
}


void init_struct(std::shared_ptr<const HipExecutor> exec,
                 std::shared_ptr<solver::SolveStruct> &solve_struct)
{
    init_struct_kernel(exec, solve_struct);
}


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const HipExecutor> exec,
              const matrix::Csr<ValueType, IndexType> *matrix,
              solver::SolveStruct *solve_struct, const gko::size_type num_rhs)
{
    generate_kernel<ValueType, IndexType>(exec, matrix, solve_struct, num_rhs,
                                          true);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_UPPER_TRS_GENERATE_KERNEL);


template <typename ValueType, typename IndexType>
void solve(std::shared_ptr<const HipExecutor> exec,
           const matrix::Csr<ValueType, IndexType> *matrix,
           const solver::SolveStruct *solve_struct,
           matrix::Dense<ValueType> *trans_b, matrix::Dense<ValueType> *trans_x,
           const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *x)
{
    solve_kernel<ValueType, IndexType>(exec, matrix, solve_struct, trans_b,
                                       trans_x, b, x);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_UPPER_TRS_SOLVE_KERNEL);


}  // namespace upper_trs
}  // namespace hip
}  // namespace kernels
}  // namespace gko
