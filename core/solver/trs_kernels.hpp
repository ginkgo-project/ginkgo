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

#ifndef GKO_CORE_SOLVER_TRS_KERNELS_HPP_
#define GKO_CORE_SOLVER_TRS_KERNELS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>

namespace gko {
namespace kernels {
namespace trs {

#define GKO_DECLARE_TRS_GENERATE_KERNEL(_vtype, _itype)        \
    void generate(std::shared_ptr<const DefaultExecutor> exec, \
                  const matrix::Csr<_vtype, _itype> *matrix,   \
                  const matrix::Dense<_vtype> *b)

#define GKO_DECLARE_TRS_SOLVE_KERNEL(_vtype, _itype)        \
    void solve(std::shared_ptr<const DefaultExecutor> exec, \
               const matrix::Csr<_vtype, _itype> *matrix,   \
               const matrix::Dense<_vtype> *b, matrix::Dense<_vtype> *x)


#define GKO_DECLARE_ALL_AS_TEMPLATES                    \
    template <typename ValueType, typename IndexType>   \
    GKO_DECLARE_TRS_SOLVE_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>   \
    GKO_DECLARE_TRS_GENERATE_KERNEL(ValueType, IndexType)


}  // namespace trs


namespace omp {
namespace trs {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace trs
}  // namespace omp


namespace cuda {
namespace trs {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace trs
}  // namespace cuda


namespace reference {
namespace trs {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace trs
}  // namespace reference


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_TRS_KERNELS_HPP
