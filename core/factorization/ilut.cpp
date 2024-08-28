// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <fstream>
#include <iostream>
#include <memory>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/factorization/ilut.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "core/base/array_access.hpp"
#include "core/base/utils.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/config/config_helper.hpp"
#include "core/factorization/factorization_kernels.hpp"
#include "core/factorization/ilut_kernels.hpp"
#include "core/factorization/par_ilut_kernels.hpp"
#include "core/matrix/coo_builder.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/matrix/csr_kernels.hpp"

namespace gko {
namespace factorization {
namespace ilut_factorization {
namespace {


GKO_REGISTER_OPERATION(convert_ptrs_to_idxs, components::convert_ptrs_to_idxs);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(csr_transpose, csr::transpose);
GKO_REGISTER_OPERATION(spgemm, csr::spgemm);
GKO_REGISTER_OPERATION(build_lookup, csr::build_lookup);
GKO_REGISTER_OPERATION(build_lookup_offsets, csr::build_lookup_offsets);
GKO_REGISTER_OPERATION(initialize_row_ptrs_l_u,
                       factorization::initialize_row_ptrs_l_u);
GKO_REGISTER_OPERATION(initialize_l_u, factorization::initialize_l_u);
GKO_REGISTER_OPERATION(add_candidates, par_ilut_factorization::add_candidates);
GKO_REGISTER_OPERATION(threshold_filter,
                       par_ilut_factorization::threshold_filter);
GKO_REGISTER_OPERATION(threshold_filter_approx,
                       par_ilut_factorization::threshold_filter_approx);
GKO_REGISTER_OPERATION(threshold_select,
                       par_ilut_factorization::threshold_select);
GKO_REGISTER_OPERATION(initialize, ilut_factorization::initialize);
GKO_REGISTER_OPERATION(compute_l_u_factors,
                       ilut_factorization::compute_l_u_factors);


}  // namespace
}  // namespace ilut_factorization

using ilut_factorization::make_add_candidates;
using ilut_factorization::make_build_lookup;
using ilut_factorization::make_build_lookup_offsets;
using ilut_factorization::make_compute_l_u_factors;
using ilut_factorization::make_convert_ptrs_to_idxs;
using ilut_factorization::make_csr_transpose;
using ilut_factorization::make_fill_array;
using ilut_factorization::make_initialize;
using ilut_factorization::make_initialize_l_u;
using ilut_factorization::make_initialize_row_ptrs_l_u;
using ilut_factorization::make_spgemm;
using ilut_factorization::make_threshold_filter;
using ilut_factorization::make_threshold_filter_approx;
using ilut_factorization::make_threshold_select;

namespace {


template <typename ValueType, typename IndexType>
struct IlutState {
    using CsrMatrix = matrix::Csr<ValueType, IndexType>;
    using CooMatrix = matrix::Coo<ValueType, IndexType>;
    using CsrBuilder = matrix::CsrBuilder<ValueType, IndexType>;
    using CooBuilder = matrix::CooBuilder<ValueType, IndexType>;
    using Scalar = matrix::Dense<ValueType>;
    // the executor on which the kernels are being executed
    std::shared_ptr<const Executor> exec;
    // max number of non-zeros L is supposed to have
    IndexType l_nnz_limit;
    // max number of non-zeros U is supposed to have
    IndexType u_nnz_limit;
    // use the approximate selection/filter kernels?
    bool use_approx_select;
    // system matrix A
    const CsrMatrix* system_matrix;
    // current lower factor L
    std::unique_ptr<CsrMatrix> l;
    // current upper factor U
    std::unique_ptr<CsrMatrix> u;
    // current upper factor U in CSC format
    std::unique_ptr<CsrMatrix> u_csc;
    // current product L * U
    std::unique_ptr<CsrMatrix> lu;
    // temporary lower factor L' before filtering
    std::unique_ptr<CsrMatrix> l_new;
    // temporary upper factor U' before filtering
    std::unique_ptr<CsrMatrix> u_new;
    // temporary upper factor U' in CSC format before filtering
    std::unique_ptr<CsrMatrix> u_new_csc;
    // lower factor L currently being updated with asynchronous iterations
    std::unique_ptr<CooMatrix> l_coo;
    // upper factor U currently being updated
    std::unique_ptr<CooMatrix> u_coo;
    // temporary array for threshold selection
    array<ValueType> selection_tmp;
    // temporary array for threshold selection
    array<remove_complex<ValueType>> selection_tmp2;
    // strategy to be used by the lower factor
    std::shared_ptr<typename CsrMatrix::strategy_type> l_strategy;
    // strategy to be used by the upper factor
    std::shared_ptr<typename CsrMatrix::strategy_type> u_strategy;

    IlutState(std::shared_ptr<const Executor> exec_in,
              const CsrMatrix* system_matrix_in,
              std::unique_ptr<CsrMatrix> l_in, std::unique_ptr<CsrMatrix> u_in,
              IndexType l_nnz_limit, IndexType u_nnz_limit,
              bool use_approx_select,
              std::shared_ptr<typename CsrMatrix::strategy_type> l_strategy_,
              std::shared_ptr<typename CsrMatrix::strategy_type> u_strategy_)
        : exec{std::move(exec_in)},
          l_nnz_limit{l_nnz_limit},
          u_nnz_limit{u_nnz_limit},
          use_approx_select{use_approx_select},
          system_matrix{system_matrix_in},
          l{std::move(l_in)},
          u{std::move(u_in)},
          selection_tmp{exec},
          selection_tmp2{exec},
          l_strategy{std::move(l_strategy_)},
          u_strategy{std::move(u_strategy_)}
    {
        auto mtx_size = system_matrix->get_size();
        auto u_nnz = u->get_num_stored_elements();
        u_csc = CsrMatrix::create(exec, mtx_size, u_nnz);
        lu = CsrMatrix::create(exec, mtx_size);
        l_new = CsrMatrix::create(exec, mtx_size);
        u_new = CsrMatrix::create(exec, mtx_size);
        u_new_csc = CsrMatrix::create(exec, mtx_size);
        l_coo = CooMatrix::create(exec, mtx_size);
        u_coo = CooMatrix::create(exec, mtx_size);
        exec->run(make_csr_transpose(u.get(), u_csc.get()));
    }

    std::unique_ptr<Composition<ValueType>> to_factors() &&
    {
        l->set_strategy(l_strategy);
        u->set_strategy(u_strategy);
        return Composition<ValueType>::create(std::move(l), std::move(u));
    }

    void iterate();
};


}  // namespace


template <typename ValueType, typename IndexType>
typename Ilut<ValueType, IndexType>::parameters_type
Ilut<ValueType, IndexType>::parse(const config::pnode& config,
                                  const config::registry& context,
                                  const config::type_descriptor& td_for_child)
{
    auto params = factorization::Ilut<ValueType, IndexType>::build();

    if (auto& obj = config.get("iterations")) {
        params.with_iterations(config::get_value<size_type>(obj));
    }
    if (auto& obj = config.get("skip_sorting")) {
        params.with_skip_sorting(config::get_value<bool>(obj));
    }
    if (auto& obj = config.get("approximate_select")) {
        params.with_approximate_select(config::get_value<bool>(obj));
    }
    if (auto& obj = config.get("deterministic_sample")) {
        params.with_deterministic_sample(config::get_value<bool>(obj));
    }
    if (auto& obj = config.get("fill_in_limit")) {
        params.with_fill_in_limit(config::get_value<double>(obj));
    }
    if (auto& obj = config.get("l_strategy")) {
        params.with_l_strategy(config::get_strategy<matrix_type>(obj));
    }
    if (auto& obj = config.get("u_strategy")) {
        params.with_u_strategy(config::get_strategy<matrix_type>(obj));
    }
    return params;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Composition<ValueType>>
Ilut<ValueType, IndexType>::generate_l_u(
    const std::shared_ptr<const LinOp>& system_matrix) const
{
    using CsrMatrix = matrix::Csr<ValueType, IndexType>;

    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);
    // make sure no invalid parameters break our kernels!
    GKO_ASSERT_EQ(parameters_.fill_in_limit > 0.0, true);

    const auto exec = this->get_executor();

    // convert and/or sort the matrix if necessary
    auto csr_system_matrix = convert_to_with_sorting<CsrMatrix>(
        exec, system_matrix, parameters_.skip_sorting);

    // initialize the L and U matrix data structures
    const auto num_rows = csr_system_matrix->get_size()[0];
    array<IndexType> l_row_ptrs_array{exec, num_rows + 1};
    array<IndexType> u_row_ptrs_array{exec, num_rows + 1};
    auto l_row_ptrs = l_row_ptrs_array.get_data();
    auto u_row_ptrs = u_row_ptrs_array.get_data();
    exec->run(make_initialize_row_ptrs_l_u(csr_system_matrix.get(), l_row_ptrs,
                                           u_row_ptrs));

    auto l_nnz =
        static_cast<size_type>(get_element(l_row_ptrs_array, num_rows));
    auto u_nnz =
        static_cast<size_type>(get_element(u_row_ptrs_array, num_rows));

    // Todo: need to set strategy of L and U?
    auto mtx_size = csr_system_matrix->get_size();
    auto l = CsrMatrix::create(exec, mtx_size, array<ValueType>{exec, l_nnz},
                               array<IndexType>{exec, l_nnz},
                               std::move(l_row_ptrs_array));
    auto u = CsrMatrix::create(exec, mtx_size, array<ValueType>{exec, u_nnz},
                               array<IndexType>{exec, u_nnz},
                               std::move(u_row_ptrs_array));

    // initialize L and U
    exec->run(make_initialize_l_u(csr_system_matrix.get(), l.get(), u.get()));
    // l->sort_by_column_index();
    //  u->sort_by_column_index(); Should be ok because system matrix is sorted?

    // setup lookup structure on factors
    array<IndexType> l_storage_offsets{exec, num_rows + 1};
    array<int64> l_row_descs{exec, num_rows};
    const auto allowed_sparsity = gko::matrix::csr::sparsity_type::bitmap |
                                  gko::matrix::csr::sparsity_type::full |
                                  gko::matrix::csr::sparsity_type::hash;
    exec->run(make_build_lookup_offsets(
        l->get_const_row_ptrs(), l->get_const_col_idxs(), num_rows,
        allowed_sparsity, l_storage_offsets.get_data()));
    const auto l_storage_size =
        static_cast<size_type>(get_element(l_storage_offsets, num_rows));
    array<int32> l_storage{exec, l_storage_size};
    exec->run(make_build_lookup(
        l->get_const_row_ptrs(), l->get_const_col_idxs(), num_rows,
        allowed_sparsity, l_storage_offsets.get_const_data(),
        l_row_descs.get_data(), l_storage.get_data()));

    array<IndexType> u_storage_offsets{exec, num_rows + 1};
    array<int64> u_row_descs{exec, num_rows};
    exec->run(make_build_lookup_offsets(
        u->get_const_row_ptrs(), u->get_const_col_idxs(), num_rows,
        allowed_sparsity, u_storage_offsets.get_data()));
    const auto u_storage_size =
        static_cast<size_type>(get_element(u_storage_offsets, num_rows));
    array<int32> u_storage{exec, u_storage_size};
    exec->run(make_build_lookup(
        u->get_const_row_ptrs(), u->get_const_col_idxs(), num_rows,
        allowed_sparsity, u_storage_offsets.get_const_data(),
        u_row_descs.get_data(), u_storage.get_data()));

    array<int> tmp{exec};
    exec->run(make_compute_l_u_factors(
        l.get(), l_storage_offsets.get_const_data(),
        l_row_descs.get_const_data(), l_storage.get_const_data(), u.get(),
        u_storage_offsets.get_const_data(), u_row_descs.get_const_data(),
        u_storage.get_const_data(), tmp));

    // compute limit #nnz for L and U
    auto l_nnz_limit =
        static_cast<IndexType>(l_nnz * parameters_.fill_in_limit);
    auto u_nnz_limit =
        static_cast<IndexType>(u_nnz * parameters_.fill_in_limit);

    IlutState<ValueType, IndexType> state{exec,
                                          csr_system_matrix.get(),
                                          std::move(l),
                                          std::move(u),
                                          l_nnz_limit,
                                          u_nnz_limit,
                                          parameters_.approximate_select,
                                          parameters_.l_strategy,
                                          parameters_.u_strategy};

    for (size_type it = 0; it < parameters_.iterations; ++it) {
        state.iterate();
    }

    return std::move(state).to_factors();
}


template <typename ValueType, typename IndexType>
void IlutState<ValueType, IndexType>::iterate()
{
    // compute L * U
    exec->run(make_spgemm(l.get(), u.get(), lu.get()));

    // add new candidates to L' and U' factors
    exec->run(make_add_candidates(lu.get(), system_matrix, l.get(), u.get(),
                                  l_new.get(), u_new.get()));

    // setup lookup structure on new factors
    const auto num_rows = system_matrix->get_size()[0];
    array<IndexType> l_new_storage_offsets{exec, num_rows + 1};
    array<int64> l_new_row_descs{exec, num_rows};
    const auto allowed_sparsity = gko::matrix::csr::sparsity_type::bitmap |
                                  gko::matrix::csr::sparsity_type::full |
                                  gko::matrix::csr::sparsity_type::hash;
    exec->run(make_build_lookup_offsets(
        l_new->get_const_row_ptrs(), l_new->get_const_col_idxs(), num_rows,
        allowed_sparsity, l_new_storage_offsets.get_data()));
    const auto l_new_storage_size =
        static_cast<size_type>(get_element(l_new_storage_offsets, num_rows));
    array<int32> l_new_storage{exec, l_new_storage_size};
    exec->run(make_build_lookup(
        l_new->get_const_row_ptrs(), l_new->get_const_col_idxs(), num_rows,
        allowed_sparsity, l_new_storage_offsets.get_const_data(),
        l_new_row_descs.get_data(), l_new_storage.get_data()));

    array<IndexType> u_new_storage_offsets{exec, num_rows + 1};
    array<int64> u_new_row_descs{exec, num_rows};
    exec->run(make_build_lookup_offsets(
        u_new->get_const_row_ptrs(), u_new->get_const_col_idxs(), num_rows,
        allowed_sparsity, u_new_storage_offsets.get_data()));
    const auto u_new_storage_size =
        static_cast<size_type>(get_element(u_new_storage_offsets, num_rows));
    array<int32> u_new_storage{exec, u_new_storage_size};
    exec->run(make_build_lookup(
        u_new->get_const_row_ptrs(), u_new->get_const_col_idxs(), num_rows,
        allowed_sparsity, u_new_storage_offsets.get_const_data(),
        u_new_row_descs.get_data(), u_new_storage.get_data()));
    // Initialize new L and U (padded with zeros)
    exec->run(make_initialize(
        system_matrix, l_new.get(), l_new_storage_offsets.get_const_data(),
        l_new_row_descs.get_const_data(), l_new_storage.get_const_data(),
        u_new.get(), u_new_storage_offsets.get_const_data(),
        u_new_row_descs.get_const_data(), u_new_storage.get_const_data()));

    // update U'(CSC), L'(COO), U'(COO) sizes and pointers
    {
        auto l_nnz = l_new->get_num_stored_elements();
        auto u_nnz = u_new->get_num_stored_elements();
        CooBuilder l_builder{l_coo};
        CooBuilder u_builder{u_coo};
        CsrBuilder u_csc_builder{u_new_csc};
        // resize arrays that will be filled
        l_builder.get_row_idx_array().resize_and_reset(l_nnz);
        u_builder.get_row_idx_array().resize_and_reset(u_nnz);
        u_csc_builder.get_col_idx_array().resize_and_reset(u_nnz);
        u_csc_builder.get_value_array().resize_and_reset(u_nnz);
        // update arrays that will be aliased
        l_builder.get_col_idx_array() =
            make_array_view(exec, l_nnz, l_new->get_col_idxs());
        u_builder.get_col_idx_array() =
            make_array_view(exec, u_nnz, u_new->get_col_idxs());
        l_builder.get_value_array() =
            make_array_view(exec, l_nnz, l_new->get_values());
        u_builder.get_value_array() =
            make_array_view(exec, u_nnz, u_new->get_values());
    }

    // convert U' into CSC format
    exec->run(make_csr_transpose(u_new.get(), u_new_csc.get()));

    // convert L' and U' into COO format
    exec->run(make_convert_ptrs_to_idxs(l_new->get_const_row_ptrs(),
                                        l_new->get_size()[0],
                                        l_coo->get_row_idxs()));
    exec->run(make_convert_ptrs_to_idxs(u_new->get_const_row_ptrs(),
                                        u_new->get_size()[0],
                                        u_coo->get_row_idxs()));

    // Factorize with new sparsity pattern
    array<int> tmp{exec};
    exec->run(make_compute_l_u_factors(
        l_new.get(), l_new_storage_offsets.get_const_data(),
        l_new_row_descs.get_const_data(), l_new_storage.get_const_data(),
        u_new.get(), u_new_storage_offsets.get_const_data(),
        u_new_row_descs.get_const_data(), u_new_storage.get_const_data(), tmp));

    // determine ranks for selection/filtering
    IndexType l_nnz = l_new->get_num_stored_elements();
    IndexType u_nnz = u_new->get_num_stored_elements();
    // make sure that the rank is in [0, *_nnz)
    auto l_filter_rank = std::max<IndexType>(0, l_nnz - l_nnz_limit - 1);
    auto u_filter_rank = std::max<IndexType>(0, u_nnz - u_nnz_limit - 1);
    remove_complex<ValueType> l_threshold{};
    remove_complex<ValueType> u_threshold{};
    CooMatrix* null_coo = nullptr;
    if (use_approx_select) {
        // remove approximately smallest candidates from L' and U'^T
        exec->run(make_threshold_filter_approx(l_new.get(), l_filter_rank,
                                               selection_tmp, l_threshold,
                                               l.get(), l_coo.get()));
        exec->run(make_threshold_filter_approx(u_new_csc.get(), u_filter_rank,
                                               selection_tmp, u_threshold,
                                               u_csc.get(), null_coo));
    } else {
        // select threshold to remove smallest candidates
        exec->run(make_threshold_select(l_new.get(), l_filter_rank,
                                        selection_tmp, selection_tmp2,
                                        l_threshold));
        exec->run(make_threshold_select(u_new_csc.get(), u_filter_rank,
                                        selection_tmp, selection_tmp2,
                                        u_threshold));

        // remove smallest candidates from L' and U'^T
        exec->run(make_threshold_filter(l_new.get(), l_threshold, l.get(),
                                        l_coo.get(), true));
        exec->run(make_threshold_filter(u_new_csc.get(), u_threshold,
                                        u_csc.get(), null_coo, true));
    }
    // remove smallest candidates from U'
    exec->run(make_threshold_filter(u_new.get(), u_threshold, u.get(),
                                    u_coo.get(), false));
}


#define GKO_DECLARE_ILUT(ValueType, IndexType) class Ilut<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ILUT);


}  // namespace factorization
}  // namespace gko
