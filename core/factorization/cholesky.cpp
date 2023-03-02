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

#include <ginkgo/core/factorization/cholesky.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "core/factorization/cholesky_kernels.hpp"
#include "core/factorization/elimination_forest.hpp"
#include "core/factorization/symbolic.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/matrix/csr_lookup.hpp"


namespace gko {
namespace experimental {
namespace factorization {
namespace {


GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(build_lookup_offsets, csr::build_lookup_offsets);
GKO_REGISTER_OPERATION(build_lookup, csr::build_lookup);
GKO_REGISTER_OPERATION(initialize, cholesky::initialize);
GKO_REGISTER_OPERATION(factorize, cholesky::factorize);


}  // namespace


template <typename ValueType, typename IndexType>
Cholesky<ValueType, IndexType>::Cholesky(std::shared_ptr<const Executor> exec,
                                         const parameters_type& params)
    : EnablePolymorphicObject<Cholesky, LinOpFactory>(std::move(exec)),
      parameters_(params)
{}


template <typename ValueType, typename IndexType>
std::unique_ptr<Factorization<ValueType, IndexType>>
Cholesky<ValueType, IndexType>::generate(
    std::shared_ptr<const LinOp> system_matrix) const
{
    auto product =
        std::unique_ptr<factorization_type>(static_cast<factorization_type*>(
            this->LinOpFactory::generate(std::move(system_matrix)).release()));
    return product;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Cholesky<ValueType, IndexType>::generate_impl(
    std::shared_ptr<const LinOp> system_matrix) const
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);
    const auto exec = this->get_executor();
    // TODO deal with non Csr matrices
    const auto mtx = as<matrix_type>(system_matrix);
    const auto num_rows = mtx->get_size()[0];
    std::unique_ptr<matrix_type> factors;
    std::unique_ptr<gko::factorization::elimination_forest<IndexType>> forest;
    if (!parameters_.symbolic_factorization) {
        gko::factorization::symbolic_cholesky(mtx.get(), factors, forest);
    } else {
        const auto& symbolic = parameters_.symbolic_factorization;
        const auto factor_nnz = symbolic->get_num_nonzeros();
        factors = matrix_type::create(exec, mtx->get_size(), factor_nnz);
        const auto symbolic_exec = symbolic->get_executor();
        exec->copy_from(symbolic_exec.get(), factor_nnz,
                        symbolic->get_const_col_idxs(),
                        factors->get_col_idxs());
        exec->copy_from(symbolic_exec.get(), num_rows + 1,
                        symbolic->get_const_row_ptrs(),
                        factors->get_row_ptrs());
        // update srow to be safe
        factors->set_strategy(factors->get_strategy());
    }
    // setup lookup structure on factors
    array<IndexType> storage_offsets{exec, num_rows + 1};
    array<int64> row_descs{exec, num_rows};
    array<IndexType> diag_idxs{exec, num_rows};
    array<IndexType> transpose_idxs{exec, factors->get_num_stored_elements()};
    const auto allowed_sparsity = gko::matrix::csr::sparsity_type::bitmap |
                                  gko::matrix::csr::sparsity_type::full |
                                  gko::matrix::csr::sparsity_type::hash;
    exec->run(make_build_lookup_offsets(
        factors->get_const_row_ptrs(), factors->get_const_col_idxs(), num_rows,
        allowed_sparsity, storage_offsets.get_data()));
    const auto storage_size = static_cast<size_type>(
        exec->copy_val_to_host(storage_offsets.get_const_data() + num_rows));
    array<int32> storage{exec, storage_size};
    exec->run(make_build_lookup(
        factors->get_const_row_ptrs(), factors->get_const_col_idxs(), num_rows,
        allowed_sparsity, storage_offsets.get_const_data(),
        row_descs.get_data(), storage.get_data()));
    // initialize factors
    exec->run(make_fill_array(factors->get_values(),
                              factors->get_num_stored_elements(),
                              zero<ValueType>()));
    exec->run(make_initialize(mtx.get(), storage_offsets.get_const_data(),
                              row_descs.get_const_data(),
                              storage.get_const_data(), diag_idxs.get_data(),
                              transpose_idxs.get_data(), factors.get()));
    // run numerical factorization
    array<int> tmp{exec};
    exec->run(make_factorize(
        storage_offsets.get_const_data(), row_descs.get_const_data(),
        storage.get_const_data(), diag_idxs.get_const_data(),
        transpose_idxs.get_const_data(), *forest, factors.get(), tmp));
    return factorization_type::create_from_combined_cholesky(
        std::move(factors));
}


#define GKO_DECLARE_CHOLESKY(ValueType, IndexType) \
    class Cholesky<ValueType, IndexType>

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CHOLESKY);


}  // namespace factorization
}  // namespace experimental
}  // namespace gko
