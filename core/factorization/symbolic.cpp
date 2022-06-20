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

#include "core/factorization/symbolic.hpp"


#include <ginkgo/core/base/temporary_clone.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>


#include "core/components/prefix_sum_kernels.hpp"
#include "core/factorization/cholesky_kernels.hpp"
#include "core/factorization/elimination_forest.hpp"
#include "core/factorization/lu_kernels.hpp"


namespace gko {
namespace factorization {
namespace {


GKO_REGISTER_OPERATION(cholesky_symbolic_count,
                       cholesky::cholesky_symbolic_count);
GKO_REGISTER_OPERATION(cholesky_symbolic,
                       cholesky::cholesky_symbolic_factorize);
GKO_REGISTER_OPERATION(prefix_sum, components::prefix_sum);
GKO_REGISTER_OPERATION(initialize, lu_factorization::initialize);
GKO_REGISTER_OPERATION(factorize, lu_factorization::factorize);


}  // namespace


/** Computes the symbolic Cholesky factorization of the given matrix. */
template <typename ValueType, typename IndexType>
std::unique_ptr<matrix::Csr<ValueType, IndexType>> symbolic_cholesky(
    const matrix::Csr<ValueType, IndexType>* mtx)
{
    using matrix_type = matrix::Csr<ValueType, IndexType>;
    const auto exec = mtx->get_executor();
    const auto host_exec = exec->get_master();
    const auto forest = compute_elim_forest(mtx);
    const auto num_rows = mtx->get_size()[0];
    array<IndexType> row_ptrs{exec, num_rows + 1};
    array<IndexType> tmp{exec};
    exec->run(
        make_cholesky_symbolic_count(mtx, forest, row_ptrs.get_data(), tmp));
    exec->run(make_prefix_sum(row_ptrs.get_data(), num_rows + 1));
    const auto factor_nnz = static_cast<size_type>(
        exec->copy_val_to_host(row_ptrs.get_const_data() + num_rows));
    auto factor = matrix_type::create(
        exec, mtx->get_size(), array<ValueType>{exec, factor_nnz},
        array<IndexType>{exec, factor_nnz}, std::move(row_ptrs));
    exec->run(make_cholesky_symbolic(mtx, forest, factor.get(), tmp));
    factor->sort_by_column_index();
    auto lt_factor = as<matrix_type>(factor->transpose());
    const auto scalar =
        initialize<matrix::Dense<ValueType>>({one<ValueType>()}, exec);
    const auto id = matrix::Identity<ValueType>::create(exec, num_rows);
    lt_factor->apply(scalar.get(), id.get(), scalar.get(), factor.get());
    return factor;
}


#define GKO_DECLARE_SYMBOLIC_CHOLESKY(ValueType, IndexType)               \
    std::unique_ptr<matrix::Csr<ValueType, IndexType>> symbolic_cholesky( \
        const matrix::Csr<ValueType, IndexType>* mtx)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SYMBOLIC_CHOLESKY);

}  // namespace factorization
}  // namespace gko
