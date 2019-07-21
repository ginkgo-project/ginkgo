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

#include <ginkgo/core/solver/trs.hpp>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include <iostream>
#include "core/matrix/csr_kernels.hpp"
#include "core/solver/trs_kernels.hpp"

namespace gko {
namespace solver {


namespace trs {


GKO_REGISTER_OPERATION(solve, trs::solve);


}  // namespace trs


template <typename ValueType, typename IndexType>
void Trs<ValueType, IndexType>::apply_impl(const LinOp *b, LinOp *x) const
{
    using Vector = matrix::Dense<ValueType>;
    using CsrMatrix = matrix::Csr<ValueType, IndexType>;

    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix_);
    const auto exec = this->get_executor();
    const auto host_exec = exec->get_master();

    auto dense_x = as<Vector>(x);
    auto dense_b = as<const Vector>(b);

    // If required, it is also possible to make this a Factory parameter
    auto csr_strategy = std::make_shared<typename CsrMatrix::cusparse>();

    // Only copies the matrix if it is not on the same executor or was not in
    // the right format. Throws an exception if it is not convertable.
    std::unique_ptr<CsrMatrix> csr_system_matrix_unique_ptr{};
    auto csr_system_matrix =
        dynamic_cast<const CsrMatrix *>(system_matrix_.get());
    if (csr_system_matrix == nullptr ||
        csr_system_matrix->get_executor() != exec) {
        csr_system_matrix_unique_ptr = CsrMatrix::create(exec);
        as<ConvertibleTo<CsrMatrix>>(system_matrix_.get())
            ->convert_to(csr_system_matrix_unique_ptr.get());
        csr_system_matrix = csr_system_matrix_unique_ptr.get();
    }
    If it needs to be sorted,
        copy it if necessary and sort it if (csr_system_matrix_unique_ptr ==
                                             nullptr)
    {
        csr_system_matrix_unique_ptr = CsrMatrix::create(exec);
        csr_system_matrix_unique_ptr->copy_from(csr_system_matrix);
    }
    csr_system_matrix_unique_ptr->sort_by_column_index();
    csr_system_matrix = csr_system_matrix_unique_ptr.get();

    exec->run(trs::make_solve(gko::lend(csr_system_matrix), dense_b, dense_x));
}


template <typename ValueType, typename IndexType>
void Trs<ValueType, IndexType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                           const LinOp *beta, LinOp *x) const
{
    auto dense_x = as<matrix::Dense<ValueType>>(x);

    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, x_clone.get());
}


#define GKO_DECLARE_TRS(_vtype, _itype) class Trs<_vtype, _itype>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_TRS);


}  // namespace solver
}  // namespace gko
