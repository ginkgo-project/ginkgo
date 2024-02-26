// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/factorization/lu.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/base/array_access.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/factorization/elimination_forest.hpp"
#include "core/factorization/lu_kernels.hpp"
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
GKO_REGISTER_OPERATION(initialize, lu_factorization::initialize);
GKO_REGISTER_OPERATION(factorize, lu_factorization::factorize);
GKO_REGISTER_HOST_OPERATION(symbolic_cholesky,
                            gko::factorization::symbolic_cholesky);
GKO_REGISTER_HOST_OPERATION(symbolic_lu, gko::factorization::symbolic_lu);
GKO_REGISTER_HOST_OPERATION(symbolic_lu_near_symm,
                            gko::factorization::symbolic_lu_near_symm);


}  // namespace


template <typename ValueType, typename IndexType>
Lu<ValueType, IndexType>::Lu(std::shared_ptr<const Executor> exec,
                             const parameters_type& params)
    : EnablePolymorphicObject<Lu, LinOpFactory>(std::move(exec)),
      parameters_(params)
{}


template <typename ValueType, typename IndexType>
std::unique_ptr<Factorization<ValueType, IndexType>>
Lu<ValueType, IndexType>::generate(
    std::shared_ptr<const LinOp> system_matrix) const
{
    auto product =
        std::unique_ptr<factorization_type>(static_cast<factorization_type*>(
            this->LinOpFactory::generate(std::move(system_matrix)).release()));
    return product;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Lu<ValueType, IndexType>::generate_impl(
    std::shared_ptr<const LinOp> system_matrix) const
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);
    const auto exec = this->get_executor();
    const auto mtx = copy_and_convert_to<matrix_type>(exec, system_matrix);
    const auto num_rows = mtx->get_size()[0];
    std::unique_ptr<matrix_type> factors;
    if (!parameters_.symbolic_factorization) {
        switch (parameters_.symbolic_algorithm) {
        case symbolic_type::general:
            exec->run(make_symbolic_lu(mtx.get(), factors));
            break;
        case symbolic_type::near_symmetric:
            exec->run(make_symbolic_lu_near_symm(mtx.get(), factors));
            break;
        case symbolic_type::symmetric: {
            std::unique_ptr<gko::factorization::elimination_forest<IndexType>>
                forest;
            exec->run(make_symbolic_cholesky(mtx.get(), true, factors, forest));
            break;
        }
        default:
            GKO_INVALID_STATE("Invalid symbolic factorization algorithm");
        }
    } else {
        const auto& symbolic = parameters_.symbolic_factorization;
        const auto factor_nnz = symbolic->get_num_nonzeros();
        factors = matrix_type::create(exec, mtx->get_size(), factor_nnz);
        const auto symbolic_exec = symbolic->get_executor();
        exec->copy_from(symbolic_exec, factor_nnz,
                        symbolic->get_const_col_idxs(),
                        factors->get_col_idxs());
        exec->copy_from(symbolic_exec, num_rows + 1,
                        symbolic->get_const_row_ptrs(),
                        factors->get_row_ptrs());
        // update srow to be safe
        factors->set_strategy(factors->get_strategy());
    }
    // setup lookup structure on factors
    array<IndexType> storage_offsets{exec, num_rows + 1};
    array<int64> row_descs{exec, num_rows};
    array<IndexType> diag_idxs{exec, num_rows};
    const auto allowed_sparsity = gko::matrix::csr::sparsity_type::bitmap |
                                  gko::matrix::csr::sparsity_type::full |
                                  gko::matrix::csr::sparsity_type::hash;
    exec->run(make_build_lookup_offsets(
        factors->get_const_row_ptrs(), factors->get_const_col_idxs(), num_rows,
        allowed_sparsity, storage_offsets.get_data()));
    const auto storage_size =
        static_cast<size_type>(get_element(storage_offsets, num_rows));
    array<int32> storage{exec, storage_size};
    exec->run(make_build_lookup(
        factors->get_const_row_ptrs(), factors->get_const_col_idxs(), num_rows,
        allowed_sparsity, storage_offsets.get_const_data(),
        row_descs.get_data(), storage.get_data()));
    // initialize factors
    exec->run(make_fill_array(factors->get_values(),
                              factors->get_num_stored_elements(),
                              zero<ValueType>()));
    exec->run(make_initialize(
        mtx.get(), storage_offsets.get_const_data(), row_descs.get_const_data(),
        storage.get_const_data(), diag_idxs.get_data(), factors.get()));
    // run numerical factorization
    array<int> tmp{exec};
    exec->run(make_factorize(storage_offsets.get_const_data(),
                             row_descs.get_const_data(),
                             storage.get_const_data(),
                             diag_idxs.get_const_data(), factors.get(), tmp));
    return factorization_type::create_from_combined_lu(std::move(factors));
}


#define GKO_DECLARE_LU(ValueType, IndexType) class Lu<ValueType, IndexType>

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_LU);


}  // namespace factorization
}  // namespace experimental
}  // namespace gko
