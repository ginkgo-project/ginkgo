// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/factorization/ilu_at.hpp"

#include <memory>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/factorization/lu.hpp>
#include <ginkgo/core/matrix/coo.hpp>

#include "core/base/array_access.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/config/config_helper.hpp"
#include "core/factorization/factorization_kernels.hpp"
#include "core/factorization/lu_kernels.hpp"
#include "core/factorization/par_ilut_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/matrix/csr_lookup.hpp"


namespace gko {
namespace factorization {
namespace ilu_factorization {
namespace {


GKO_REGISTER_OPERATION(spgemm, csr::spgemm);
GKO_REGISTER_OPERATION(add_diagonal_elements,
                       factorization::add_diagonal_elements);
GKO_REGISTER_OPERATION(initialize_row_ptrs_l_u,
                       factorization::initialize_row_ptrs_l_u);
GKO_REGISTER_OPERATION(initialize_l_u, factorization::initialize_l_u);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(initialize, lu_factorization::initialize);
GKO_REGISTER_OPERATION(factorize, lu_factorization::factorize);
GKO_REGISTER_OPERATION(threshold_select,
                       par_ilut_factorization::threshold_select);
GKO_REGISTER_OPERATION(threshold_filter,
                       par_ilut_factorization::threshold_filter);


}  // anonymous namespace
}  // namespace ilu_factorization


template <typename ValueType, typename IndexType>
typename IluAt<ValueType, IndexType>::parameters_type
IluAt<ValueType, IndexType>::parse(const config::pnode& config,
                                   const config::registry& context,
                                   const config::type_descriptor& td_for_child)
{
    auto params = factorization::IluAt<ValueType, IndexType>::build();
    if (auto& obj = config.get("l_strategy")) {
        params.with_l_strategy(config::get_strategy<matrix_type>(obj));
    }
    if (auto& obj = config.get("u_strategy")) {
        params.with_u_strategy(config::get_strategy<matrix_type>(obj));
    }
    if (auto& obj = config.get("skip_sorting")) {
        params.with_skip_sorting(config::get_value<bool>(obj));
    }
    if (auto& obj = config.get("fill_in_limit")) {
        params.with_fill_in_limit(config::get_value<double>(obj));
    }
    if (auto& obj = config.get("superlevel")) {
        params.with_superlevel(config::get_value<int32>(obj));
    }
    return params;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Composition<ValueType>>
IluAt<ValueType, IndexType>::generate_l_u(
    const std::shared_ptr<const LinOp>& system_matrix, bool skip_sorting) const
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);

    const auto exec = this->get_executor();

    // Converts the system matrix to CSR.
    // Throws an exception if it is not convertible.
    auto local_system_matrix = share(matrix_type::create(exec));
    as<ConvertibleTo<matrix_type>>(system_matrix.get())
        ->convert_to(local_system_matrix);

    if (!skip_sorting) {
        local_system_matrix->sort_by_column_index();
    }

    // Add explicit diagonal zero elements if they are missing
    exec->run(ilu_factorization::make_add_diagonal_elements(
        local_system_matrix.get(), false));


    // Set initial sparsity pattern using "superlevel" (will include
    // all elements of standard level, plus some from higher levels).
    exec->synchronize();
    auto tic = std::chrono::high_resolution_clock::now();

    // Start with the same sparsity pattern as the system matrix
    const auto nnz = local_system_matrix->get_num_stored_elements();
    const auto num_rows = local_system_matrix->get_size()[0];
    array<SparsityType> sp_ones{exec, nnz};
    sp_ones.fill(one<SparsityType>());
    array<IndexType> sp_col_idxs{exec, nnz};
    array<IndexType> sp_row_ptrs{exec, num_rows + 1};
    exec->copy_from(exec, nnz, local_system_matrix->get_const_col_idxs(),
                    sp_col_idxs.get_data());
    exec->copy_from(exec, num_rows + 1,
                    local_system_matrix->get_const_row_ptrs(),
                    sp_row_ptrs.get_data());
    auto sparsity = share(
        sparsity_matrix_type::create(exec, local_system_matrix->get_size(),
                                     sp_ones, sp_col_idxs, sp_row_ptrs));
    size_type l_factor_nnz_limit;
    size_type u_factor_nnz_limit;
    const auto matrix_size = local_system_matrix->get_size();
    // If not ILU0, continue building sparsity pattern
    if (parameters_.superlevel > 0) {
        // Recursively form L and U and do L * U to move to next superlevel
        // Separate L and U factors: nnz
        array<IndexType> new_l_row_ptrs{exec, num_rows + 1};
        array<IndexType> new_u_row_ptrs{exec, num_rows + 1};
        array<IndexType> new_l_col_idxs{exec};
        array<SparsityType> new_l_vals{exec};
        array<IndexType> new_u_col_idxs{exec};
        array<SparsityType> new_u_vals{exec};

        size_type new_l_nnz;
        size_type new_u_nnz;

        std::shared_ptr<sparsity_matrix_type> new_l_factor;
        std::shared_ptr<sparsity_matrix_type> new_u_factor;
        for (int l = 0; l < parameters_.superlevel; l++) {
            // Split into L and U
            exec->run(ilu_factorization::make_initialize_row_ptrs_l_u(
                sparsity.get(), new_l_row_ptrs.get_data(),
                new_u_row_ptrs.get_data()));

            // Get nnz from device memory
            new_l_nnz =
                static_cast<size_type>(get_element(new_l_row_ptrs, num_rows));
            new_u_nnz =
                static_cast<size_type>(get_element(new_u_row_ptrs, num_rows));

            // If level = 0, use the sizes to set the fill-in for the final
            // factors
            if (l == 0) {
                l_factor_nnz_limit = parameters_.fill_in_limit * new_l_nnz;
                u_factor_nnz_limit = parameters_.fill_in_limit * new_u_nnz;
            }

            // Init arrays
            new_l_col_idxs.resize_and_reset(new_l_nnz);
            new_l_vals.resize_and_reset(new_l_nnz);
            new_u_col_idxs.resize_and_reset(new_u_nnz);
            new_u_vals.resize_and_reset(new_u_nnz);

            new_l_factor = sparsity_matrix_type::create(
                exec, matrix_size, new_l_vals.as_view(),
                new_l_col_idxs.as_view(), new_l_row_ptrs.as_view());
            new_u_factor = sparsity_matrix_type::create(
                exec, matrix_size, new_u_vals.as_view(),
                new_u_col_idxs.as_view(), new_u_row_ptrs.as_view());

            exec->run(ilu_factorization::make_initialize_l_u(
                sparsity.get(), new_l_factor.get(), new_u_factor.get()));

            // sparsity_{i+1} = L_{i} * U_{i}
            exec->run(ilu_factorization::make_spgemm(
                new_l_factor.get(), new_u_factor.get(), sparsity.get()));
        }
    }

    array<ValueType> factors_vals{exec, sparsity->get_num_stored_elements()};
    array<IndexType> new_col_idxs{exec, sparsity->get_num_stored_elements()};
    exec->copy_from(exec, sparsity->get_num_stored_elements(),
                    sparsity->get_const_col_idxs(), new_col_idxs.get_data());
    array<IndexType> new_row_ptrs{exec, num_rows + 1};
    exec->copy_from(exec, num_rows + 1, sparsity->get_const_row_ptrs(),
                    new_row_ptrs.get_data());
    auto factors = share(matrix_type::create(
        exec, local_system_matrix->get_size(), factors_vals, new_col_idxs,
        new_row_ptrs, local_system_matrix->get_strategy()));
    factors->sort_by_column_index();
    exec->synchronize();
    auto toc = std::chrono::high_resolution_clock::now();
    auto sparsity_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);
    std::cout << "Time to create sparsity pattern: " << sparsity_time.count()
              << std::endl;

    exec->synchronize();
    tic = std::chrono::high_resolution_clock::now();
    // setup lookup structure on factors
    const auto lookup = matrix::csr::build_lookup(factors.get());
    array<IndexType> diag_idxs{exec, num_rows};
    exec->run(ilu_factorization::make_initialize(
        local_system_matrix.get(), lookup.storage_offsets.get_const_data(),
        lookup.row_descs.get_const_data(), lookup.storage.get_const_data(),
        diag_idxs.get_data(), factors.get()));
    // run numerical factorization
    array<int> tmp{exec};
    exec->run(ilu_factorization::make_factorize(
        lookup.storage_offsets.get_const_data(),
        lookup.row_descs.get_const_data(), lookup.storage.get_const_data(),
        diag_idxs.get_const_data(), factors.get(), false, tmp));

    // Separate L and U factors: nnz
    array<IndexType> l_row_ptrs{exec, num_rows + 1};
    array<IndexType> u_row_ptrs{exec, num_rows + 1};
    exec->run(ilu_factorization::make_initialize_row_ptrs_l_u(
        factors.get(), l_row_ptrs.get_data(), u_row_ptrs.get_data()));

    // Get nnz from device memory
    auto l_nnz = static_cast<size_type>(get_element(l_row_ptrs, num_rows));
    auto u_nnz = static_cast<size_type>(get_element(u_row_ptrs, num_rows));
    if (parameters_.superlevel == 0) {
        l_factor_nnz_limit = parameters_.fill_in_limit * l_nnz;
        u_factor_nnz_limit = parameters_.fill_in_limit * u_nnz;
    }

    // Init arrays
    array<IndexType> l_col_idxs{exec, l_nnz};
    array<ValueType> l_vals{exec, l_nnz};
    std::shared_ptr<matrix_type> l_factor = matrix_type::create(
        exec, matrix_size, std::move(l_vals), std::move(l_col_idxs),
        std::move(l_row_ptrs), parameters_.l_strategy);
    array<IndexType> u_col_idxs{exec, u_nnz};
    array<ValueType> u_vals{exec, u_nnz};
    std::shared_ptr<matrix_type> u_factor = matrix_type::create(
        exec, matrix_size, std::move(u_vals), std::move(u_col_idxs),
        std::move(u_row_ptrs), parameters_.u_strategy);

    // Separate L and U: columns and values
    exec->run(ilu_factorization::make_initialize_l_u(
        factors.get(), l_factor.get(), u_factor.get()));

    exec->synchronize();
    toc = std::chrono::high_resolution_clock::now();
    auto factor_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);
    std::cout << "Time to factorize: " << factor_time.count() << std::endl;

    std::cout << "nnz before dropping: L, U = " << l_nnz << ", " << u_nnz
              << std::endl;
    // Select dropping thresholds and drop values from L and U
    exec->synchronize();
    tic = std::chrono::high_resolution_clock::now();

    matrix::Coo<ValueType, IndexType>* null_coo = nullptr;
    auto l_filter_rank = std::max<IndexType>(0, l_nnz - l_factor_nnz_limit - 1);
    auto u_filter_rank = std::max<IndexType>(0, u_nnz - u_factor_nnz_limit - 1);
    remove_complex<ValueType> l_threshold{};
    remove_complex<ValueType> u_threshold{};
    array<ValueType> selection_tmp{exec};
    array<remove_complex<ValueType>> selection_tmp2{exec};

    if (l_filter_rank > 0) {
        exec->run(ilu_factorization::make_threshold_select(
            l_factor.get(), l_filter_rank, selection_tmp, selection_tmp2,
            l_threshold));
        std::shared_ptr<matrix_type> l_factor_post_drop = clone(exec, l_factor);
        std::cout << "dropping from L with threshold " << l_threshold
                  << std::endl;
        exec->run(ilu_factorization::make_threshold_filter(
            l_factor.get(), l_threshold, l_factor_post_drop.get(), null_coo,
            true));
        l_factor = std::move(l_factor_post_drop);
    }
    if (u_filter_rank > 0) {
        exec->run(ilu_factorization::make_threshold_select(
            u_factor.get(), u_filter_rank, selection_tmp, selection_tmp2,
            u_threshold));
        std::shared_ptr<matrix_type> u_factor_post_drop = clone(exec, u_factor);
        std::cout << "dropping from U with threshold " << u_threshold
                  << std::endl;
        exec->run(ilu_factorization::make_threshold_filter(
            u_factor.get(), u_threshold, u_factor_post_drop.get(), null_coo,
            false));
        u_factor = std::move(u_factor_post_drop);
    }

    std::cout << "final nnz: L, U = " << l_factor->get_num_stored_elements()
              << ", " << u_factor->get_num_stored_elements() << std::endl;
    exec->synchronize();
    toc = std::chrono::high_resolution_clock::now();
    auto drop_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);
    std::cout << "Time to drop: " << drop_time.count() << std::endl;
    return Composition<ValueType>::create(std::move(l_factor),
                                          std::move(u_factor));
}


#define GKO_DECLARE_ILUAT(ValueType, IndexType) \
    class IluAt<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ILUAT);


}  // namespace factorization
}  // namespace gko
