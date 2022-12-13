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

#include <ginkgo/core/preconditioner/schwarz.hpp>


#include <memory>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/extended_float.hpp"
#include "core/base/utils.hpp"
#include "core/distributed/helpers.hpp"


namespace gko {
namespace preconditioner {


template <typename ValueType, typename IndexType, typename GlobalIndexType>
void Schwarz<ValueType, IndexType, GlobalIndexType>::apply_impl(const LinOp* b,
                                                                LinOp* x) const
{
    experimental::precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x);
        },
        b, x);
}


template <typename ValueType, typename IndexType, typename GlobalIndexType>
template <typename VectorType>
void Schwarz<ValueType, IndexType, GlobalIndexType>::apply_dense_impl(
    const VectorType* dense_b, VectorType* dense_x) const
{
    using LocalVector = matrix::Dense<ValueType>;
    if (is_distributed()) {
        GKO_ASSERT(this->inner_solvers_.size() > 0);
        this->inner_solvers_[0]->apply(detail::get_local(dense_b),
                                       detail::get_local(dense_x));
    }
}


template <typename ValueType, typename IndexType, typename GlobalIndexType>
void Schwarz<ValueType, IndexType, GlobalIndexType>::apply_impl(
    const LinOp* alpha, const LinOp* b, const LinOp* beta, LinOp* x) const
{
    experimental::precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_dense_impl(dense_b, x_clone.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, x_clone.get());
        },
        alpha, b, beta, x);
}


template <typename ValueType, typename IndexType, typename GlobalIndexType>
std::unique_ptr<LinOp>
Schwarz<ValueType, IndexType, GlobalIndexType>::transpose() const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType, typename GlobalIndexType>
std::unique_ptr<LinOp>
Schwarz<ValueType, IndexType, GlobalIndexType>::conj_transpose() const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType, typename GlobalIndexType>
void Schwarz<ValueType, IndexType, GlobalIndexType>::generate(
    const LinOp* system_matrix)
{
    using base_mat = matrix::Csr<ValueType, IndexType>;
#if GINKGO_BUILD_MPI
    using dist_mat = experimental::distributed::Matrix<ValueType, IndexType,
                                                       GlobalIndexType>;
#endif
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);
    if (dynamic_cast<const base_mat*>(system_matrix) != nullptr) {
        GKO_NOT_IMPLEMENTED;
        this->is_distributed_ = false;
#if GINKGO_BUILD_MPI
    } else if (dynamic_cast<const dist_mat*>(system_matrix) != nullptr) {
        auto mat = as<const dist_mat>(system_matrix);
        if (mat->get_executor() != this->get_executor()) {
            GKO_NOT_IMPLEMENTED;
        }

        if (parameters_.inner_solver) {
            this->inner_solvers_.emplace_back(
                parameters_.inner_solver->generate(mat->get_local_matrix()));
        }
        if (this->inner_solvers_.size() < 1) {
            GKO_NOT_IMPLEMENTED;
        }
        this->is_distributed_ = true;
#endif
    }
}


#define GKO_DECLARE_SCHWARZ(ValueType, IndexType, GlobalIndexType) \
    class Schwarz<ValueType, IndexType, GlobalIndexType>
GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_SCHWARZ);


}  // namespace preconditioner
}  // namespace gko
