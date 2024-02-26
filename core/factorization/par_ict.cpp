// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/factorization/par_ict.hpp>


#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/base/array_access.hpp"
#include "core/base/utils.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/factorization/factorization_kernels.hpp"
#include "core/factorization/par_ict_kernels.hpp"
#include "core/factorization/par_ilu_kernels.hpp"
#include "core/factorization/par_ilut_kernels.hpp"
#include "core/matrix/coo_builder.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/matrix/csr_kernels.hpp"


namespace gko {
namespace factorization {
namespace par_ict_factorization {
namespace {


GKO_REGISTER_OPERATION(threshold_select,
                       par_ilut_factorization::threshold_select);
GKO_REGISTER_OPERATION(threshold_filter,
                       par_ilut_factorization::threshold_filter);
GKO_REGISTER_OPERATION(threshold_filter_approx,
                       par_ilut_factorization::threshold_filter_approx);
GKO_REGISTER_OPERATION(add_candidates, par_ict_factorization::add_candidates);
GKO_REGISTER_OPERATION(compute_factor, par_ict_factorization::compute_factor);

GKO_REGISTER_OPERATION(initialize_row_ptrs_l,
                       factorization::initialize_row_ptrs_l);
GKO_REGISTER_OPERATION(initialize_l, factorization::initialize_l);

GKO_REGISTER_OPERATION(csr_conj_transpose, csr::conj_transpose);
GKO_REGISTER_OPERATION(convert_ptrs_to_idxs, components::convert_ptrs_to_idxs);
GKO_REGISTER_OPERATION(spgemm, csr::spgemm);


}  // namespace
}  // namespace par_ict_factorization


using par_ict_factorization::make_add_candidates;
using par_ict_factorization::make_compute_factor;
using par_ict_factorization::make_convert_ptrs_to_idxs;
using par_ict_factorization::make_csr_conj_transpose;
using par_ict_factorization::make_initialize_l;
using par_ict_factorization::make_initialize_row_ptrs_l;
using par_ict_factorization::make_spgemm;
using par_ict_factorization::make_threshold_filter;
using par_ict_factorization::make_threshold_filter_approx;
using par_ict_factorization::make_threshold_select;


namespace {


template <typename ValueType, typename IndexType>
struct ParIctState {
    using CsrMatrix = matrix::Csr<ValueType, IndexType>;
    using CooMatrix = matrix::Coo<ValueType, IndexType>;
    using CsrBuilder = matrix::CsrBuilder<ValueType, IndexType>;
    using CooBuilder = matrix::CooBuilder<ValueType, IndexType>;
    using Scalar = matrix::Dense<ValueType>;
    // the executor on which the kernels are being executed
    std::shared_ptr<const Executor> exec;
    // max number of non-zeros L is supposed to have
    IndexType l_nnz_limit;
    // use the approximate selection/filter kernels?
    bool use_approx_select;
    // system matrix A
    const CsrMatrix* system_matrix;
    // current lower factor L
    std::unique_ptr<CsrMatrix> l;
    // current upper factor L^H
    std::unique_ptr<CsrMatrix> lh;
    // current product L * L^H
    std::unique_ptr<CsrMatrix> llh;
    // temporary lower factor L' before filtering
    std::unique_ptr<CsrMatrix> l_new;
    // lower factor L currently being updated with asynchronous iterations
    std::unique_ptr<CooMatrix> l_coo;
    // temporary array for threshold selection
    array<ValueType> selection_tmp;
    // temporary array for threshold selection
    array<remove_complex<ValueType>> selection_tmp2;
    // strategy to be used by the lower factor
    std::shared_ptr<typename CsrMatrix::strategy_type> l_strategy;
    // strategy to be used by the upper factor
    std::shared_ptr<typename CsrMatrix::strategy_type> lh_strategy;

    ParIctState(std::shared_ptr<const Executor> exec_in,
                const CsrMatrix* system_matrix_in,
                std::unique_ptr<CsrMatrix> l_in, IndexType l_nnz_limit,
                bool use_approx_select,
                std::shared_ptr<typename CsrMatrix::strategy_type> l_strategy_,
                std::shared_ptr<typename CsrMatrix::strategy_type> lh_strategy_)
        : exec{std::move(exec_in)},
          l_nnz_limit{l_nnz_limit},
          use_approx_select{use_approx_select},
          system_matrix{system_matrix_in},
          l{std::move(l_in)},
          selection_tmp{exec},
          selection_tmp2{exec},
          l_strategy{std::move(l_strategy_)},
          lh_strategy{std::move(lh_strategy_)}
    {
        auto mtx_size = system_matrix->get_size();
        auto l_nnz = l->get_num_stored_elements();
        lh = CsrMatrix::create(exec, mtx_size, l_nnz);
        llh = CsrMatrix::create(exec, mtx_size);
        l_new = CsrMatrix::create(exec, mtx_size);
        l_coo = CooMatrix::create(exec, mtx_size);
        exec->run(make_csr_conj_transpose(l.get(), lh.get()));
    }

    std::unique_ptr<Composition<ValueType>> to_factors() &&
    {
        l->set_strategy(l_strategy);
        lh->set_strategy(lh_strategy);
        return Composition<ValueType>::create(std::move(l), std::move(lh));
    }

    void iterate();
};


}  // namespace


template <typename ValueType, typename IndexType>
std::unique_ptr<Composition<ValueType>>
ParIct<ValueType, IndexType>::generate_l_lt(
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

    // initialize the L matrix data structures
    const auto num_rows = csr_system_matrix->get_size()[0];
    array<IndexType> l_row_ptrs_array{exec, num_rows + 1};
    auto l_row_ptrs = l_row_ptrs_array.get_data();
    exec->run(make_initialize_row_ptrs_l(csr_system_matrix.get(), l_row_ptrs));

    auto l_nnz =
        static_cast<size_type>(get_element(l_row_ptrs_array, num_rows));

    auto mtx_size = csr_system_matrix->get_size();
    auto l = CsrMatrix::create(exec, mtx_size, array<ValueType>{exec, l_nnz},
                               array<IndexType>{exec, l_nnz},
                               std::move(l_row_ptrs_array));

    // initialize L
    exec->run(make_initialize_l(csr_system_matrix.get(), l.get(), true));

    // compute limit #nnz for L
    auto l_nnz_limit =
        static_cast<IndexType>(l_nnz * parameters_.fill_in_limit);

    ParIctState<ValueType, IndexType> state{exec,
                                            csr_system_matrix.get(),
                                            std::move(l),
                                            l_nnz_limit,
                                            parameters_.approximate_select,
                                            parameters_.l_strategy,
                                            parameters_.lt_strategy};

    for (size_type it = 0; it < parameters_.iterations; ++it) {
        state.iterate();
    }

    return std::move(state).to_factors();
}


template <typename ValueType, typename IndexType>
void ParIctState<ValueType, IndexType>::iterate()
{
    // compute L * L^H
    exec->run(make_spgemm(l.get(), lh.get(), llh.get()));

    // add new candidates to L' factor
    exec->run(
        make_add_candidates(llh.get(), system_matrix, l.get(), l_new.get()));

    // update L(COO), L'^H sizes and pointers
    {
        auto l_nnz = l_new->get_num_stored_elements();
        CooBuilder l_builder{l_coo};
        // resize arrays that will be filled
        l_builder.get_row_idx_array().resize_and_reset(l_nnz);
        // update arrays that will be aliased
        l_builder.get_col_idx_array() =
            make_array_view(exec, l_nnz, l_new->get_col_idxs());
        l_builder.get_value_array() =
            make_array_view(exec, l_nnz, l_new->get_values());
    }

    // convert L into COO format
    exec->run(make_convert_ptrs_to_idxs(l_new->get_const_row_ptrs(),
                                        l_new->get_size()[0],
                                        l_coo->get_row_idxs()));

    // execute asynchronous iteration
    exec->run(make_compute_factor(system_matrix, l_new.get(), l_coo.get()));

    // determine ranks for selection/filtering
    IndexType l_nnz = l_new->get_num_stored_elements();
    // make sure that the rank is in [0, *_nnz)
    auto l_filter_rank = std::max<IndexType>(0, l_nnz - l_nnz_limit - 1);
    if (use_approx_select) {
        remove_complex<ValueType> tmp{};
        // remove approximately smallest candidates
        exec->run(make_threshold_filter_approx(l_new.get(), l_filter_rank,
                                               selection_tmp, tmp, l.get(),
                                               l_coo.get()));
    } else {
        // select threshold to remove smallest candidates
        remove_complex<ValueType> l_threshold{};
        exec->run(make_threshold_select(l_new.get(), l_filter_rank,
                                        selection_tmp, selection_tmp2,
                                        l_threshold));

        // remove smallest candidates
        exec->run(make_threshold_filter(l_new.get(), l_threshold, l.get(),
                                        l_coo.get(), true));
    }

    // execute asynchronous iteration
    exec->run(make_compute_factor(system_matrix, l.get(), l_coo.get()));

    // convert L to L^H
    {
        auto l_nnz = l->get_num_stored_elements();
        CsrBuilder lt_builder{lh};
        lt_builder.get_col_idx_array().resize_and_reset(l_nnz);
        lt_builder.get_value_array().resize_and_reset(l_nnz);
    }
    exec->run(make_csr_conj_transpose(l.get(), lh.get()));
}


#define GKO_DECLARE_PAR_ICT(ValueType, IndexType) \
    class ParIct<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PAR_ICT);


}  // namespace factorization
}  // namespace gko
