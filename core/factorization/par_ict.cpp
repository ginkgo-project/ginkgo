/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include <ginkgo/core/factorization/par_ict.hpp>


#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>


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
GKO_REGISTER_OPERATION(convert_to_coo, csr::convert_to_coo);
GKO_REGISTER_OPERATION(spgemm, csr::spgemm);


}  // namespace par_ict_factorization


using par_ict_factorization::make_add_candidates;
using par_ict_factorization::make_compute_factor;
using par_ict_factorization::make_convert_to_coo;
using par_ict_factorization::make_csr_conj_transpose;
using par_ict_factorization::make_initialize_l;
using par_ict_factorization::make_initialize_row_ptrs_l;
using par_ict_factorization::make_spgemm;
using par_ict_factorization::make_threshold_filter;
using par_ict_factorization::make_threshold_filter_approx;
using par_ict_factorization::make_threshold_select;


template <typename ValueType, typename IndexType>
class ParIctState {
    friend class ParIct<ValueType, IndexType>;

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
    const CsrMatrix *system_matrix;
    // current lower factor L
    std::unique_ptr<CsrMatrix> l;
    // current upper factor L^H
    std::unique_ptr<CsrMatrix> lt;
    // current product L * L^H
    std::unique_ptr<CsrMatrix> llt;
    // temporary lower factor L' before filtering
    std::unique_ptr<CsrMatrix> l_new;
    // lower factor L currently being updated with asynchronous iterations
    std::unique_ptr<CooMatrix> l_coo;
    // temporary array for threshold selection
    Array<ValueType> selection_tmp;
    // temporary array for threshold selection
    Array<remove_complex<ValueType>> selection_tmp2;
    // strategy to be used by the lower factor
    std::shared_ptr<typename CsrMatrix::strategy_type> l_strategy;
    // strategy to be used by the upper factor
    std::shared_ptr<typename CsrMatrix::strategy_type> lt_strategy;

    ParIctState(std::shared_ptr<const Executor> exec_in,
                const CsrMatrix *system_matrix_in,
                std::unique_ptr<CsrMatrix> l_in, IndexType l_nnz_limit,
                bool use_approx_select,
                std::shared_ptr<typename CsrMatrix::strategy_type> l_strategy_,
                std::shared_ptr<typename CsrMatrix::strategy_type> lt_strategy_)
        : exec{std::move(exec_in)},
          l_nnz_limit{l_nnz_limit},
          use_approx_select{use_approx_select},
          system_matrix{system_matrix_in},
          l{std::move(l_in)},
          selection_tmp{exec},
          selection_tmp2{exec},
          l_strategy{std::move(l_strategy_)},
          lt_strategy{std::move(lt_strategy_)}
    {
        auto mtx_size = system_matrix->get_size();
        auto l_nnz = l->get_num_stored_elements();
        lt = CsrMatrix::create(exec, mtx_size, l_nnz);
        llt = CsrMatrix::create(exec, mtx_size);
        l_new = CsrMatrix::create(exec, mtx_size);
        l_coo = CooMatrix::create(exec, mtx_size);
        exec->run(make_csr_conj_transpose(l.get(), lt.get()));
    }

    std::unique_ptr<Composition<ValueType>> to_factors() &&
    {
        l->set_strategy(l_strategy);
        lt->set_strategy(lt_strategy);
        return Composition<ValueType>::create(std::move(l), std::move(lt));
    }

    void iterate();
};


template <typename ValueType, typename IndexType>
std::unique_ptr<Composition<ValueType>>
ParIct<ValueType, IndexType>::generate_l_lt(
    const std::shared_ptr<const LinOp> &system_matrix) const
{
    using CsrMatrix = matrix::Csr<ValueType, IndexType>;

    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);
    // make sure no invalid parameters break our kernels!
    GKO_ASSERT_EQ(parameters_.fill_in_limit > 0.0, true);

    const auto exec = this->get_executor();

    // convert and/or sort the matrix if necessary
    std::unique_ptr<CsrMatrix> csr_system_matrix_unique_ptr{};
    auto csr_system_matrix =
        dynamic_cast<const CsrMatrix *>(system_matrix.get());
    if (csr_system_matrix == nullptr ||
        csr_system_matrix->get_executor() != exec) {
        csr_system_matrix_unique_ptr = CsrMatrix::create(exec);
        as<ConvertibleTo<CsrMatrix>>(system_matrix.get())
            ->convert_to(csr_system_matrix_unique_ptr.get());
        csr_system_matrix = csr_system_matrix_unique_ptr.get();
    }
    if (!parameters_.skip_sorting) {
        if (csr_system_matrix_unique_ptr == nullptr) {
            csr_system_matrix_unique_ptr = CsrMatrix::create(exec);
            csr_system_matrix_unique_ptr->copy_from(csr_system_matrix);
        }
        csr_system_matrix_unique_ptr->sort_by_column_index();
        csr_system_matrix = csr_system_matrix_unique_ptr.get();
    }

    // initialize the L matrix data structures
    const auto num_rows = csr_system_matrix->get_size()[0];
    Array<IndexType> l_row_ptrs_array{exec, num_rows + 1};
    auto l_row_ptrs = l_row_ptrs_array.get_data();
    exec->run(make_initialize_row_ptrs_l(csr_system_matrix, l_row_ptrs));

    auto l_nnz =
        static_cast<size_type>(exec->copy_val_to_host(l_row_ptrs + num_rows));

    auto mtx_size = csr_system_matrix->get_size();
    auto l = CsrMatrix::create(exec, mtx_size, Array<ValueType>{exec, l_nnz},
                               Array<IndexType>{exec, l_nnz},
                               std::move(l_row_ptrs_array));

    // initialize L
    exec->run(make_initialize_l(csr_system_matrix, l.get(), true));

    // compute limit #nnz for L
    auto l_nnz_limit =
        static_cast<IndexType>(l_nnz * parameters_.fill_in_limit);

    ParIctState<ValueType, IndexType> state{exec,
                                            csr_system_matrix,
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
    exec->run(make_spgemm(l.get(), lt.get(), llt.get()));

    // add new candidates to L' factor
    exec->run(
        make_add_candidates(llt.get(), system_matrix, l.get(), l_new.get()));

    // update L(COO), L'^H sizes and pointers
    {
        auto l_nnz = l_new->get_num_stored_elements();
        CooBuilder l_builder{l_coo.get()};
        // resize arrays that will be filled
        l_builder.get_row_idx_array().resize_and_reset(l_nnz);
        // update arrays that will be aliased
        l_builder.get_col_idx_array() =
            Array<IndexType>::view(exec, l_nnz, l_new->get_col_idxs());
        l_builder.get_value_array() =
            Array<ValueType>::view(exec, l_nnz, l_new->get_values());
    }

    // convert L into COO format
    exec->run(make_convert_to_coo(l_new.get(), l_coo.get()));

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
        CsrBuilder lt_builder{lt.get()};
        lt_builder.get_col_idx_array().resize_and_reset(l_nnz);
        lt_builder.get_value_array().resize_and_reset(l_nnz);
    }
    exec->run(make_csr_conj_transpose(l.get(), lt.get()));
}


#define GKO_DECLARE_PAR_ICT(ValueType, IndexType) \
    class ParIct<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PAR_ICT);


}  // namespace factorization
}  // namespace gko