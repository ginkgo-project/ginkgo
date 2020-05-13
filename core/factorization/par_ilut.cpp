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

#include <ginkgo/core/factorization/par_ilut.hpp>


#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/factorization/factorization_kernels.hpp"
#include "core/factorization/par_ilu_kernels.hpp"
#include "core/factorization/par_ilut_kernels.hpp"
#include "core/matrix/coo_builder.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/matrix/csr_kernels.hpp"


namespace gko {
namespace factorization {
namespace par_ilut_factorization {


GKO_REGISTER_OPERATION(threshold_select,
                       par_ilut_factorization::threshold_select);
GKO_REGISTER_OPERATION(threshold_filter,
                       par_ilut_factorization::threshold_filter);
GKO_REGISTER_OPERATION(threshold_filter_approx,
                       par_ilut_factorization::threshold_filter_approx);
GKO_REGISTER_OPERATION(add_candidates, par_ilut_factorization::add_candidates);
GKO_REGISTER_OPERATION(compute_l_u_factors,
                       par_ilut_factorization::compute_l_u_factors);

GKO_REGISTER_OPERATION(initialize_row_ptrs_l_u,
                       factorization::initialize_row_ptrs_l_u);
GKO_REGISTER_OPERATION(initialize_l_u, factorization::initialize_l_u);

GKO_REGISTER_OPERATION(csr_transpose, csr::transpose);
GKO_REGISTER_OPERATION(convert_to_coo, csr::convert_to_coo);
GKO_REGISTER_OPERATION(spgemm, csr::spgemm);


}  // namespace par_ilut_factorization


using par_ilut_factorization::make_add_candidates;
using par_ilut_factorization::make_compute_l_u_factors;
using par_ilut_factorization::make_convert_to_coo;
using par_ilut_factorization::make_csr_transpose;
using par_ilut_factorization::make_initialize_l_u;
using par_ilut_factorization::make_initialize_row_ptrs_l_u;
using par_ilut_factorization::make_spgemm;
using par_ilut_factorization::make_threshold_filter;
using par_ilut_factorization::make_threshold_filter_approx;
using par_ilut_factorization::make_threshold_select;


template <typename ValueType, typename IndexType>
class ParIlutState {
    friend class ParIlut<ValueType, IndexType>;

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
    const CsrMatrix *system_matrix;
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
    Array<ValueType> selection_tmp;
    // temporary array for threshold selection
    Array<remove_complex<ValueType>> selection_tmp2;
    // strategy to be used by the lower factor
    std::shared_ptr<typename CsrMatrix::strategy_type> l_strategy;
    // strategy to be used by the upper factor
    std::shared_ptr<typename CsrMatrix::strategy_type> u_strategy;

    ParIlutState(std::shared_ptr<const Executor> exec_in,
                 const CsrMatrix *system_matrix_in,
                 std::unique_ptr<CsrMatrix> l_in,
                 std::unique_ptr<CsrMatrix> u_in, IndexType l_nnz_limit,
                 IndexType u_nnz_limit, bool use_approx_select,
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


template <typename ValueType, typename IndexType>
std::unique_ptr<Composition<ValueType>>
ParIlut<ValueType, IndexType>::generate_l_u(
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

    // initialize the L and U matrix data structures
    const auto num_rows = csr_system_matrix->get_size()[0];
    Array<IndexType> l_row_ptrs_array{exec, num_rows + 1};
    Array<IndexType> u_row_ptrs_array{exec, num_rows + 1};
    auto l_row_ptrs = l_row_ptrs_array.get_data();
    auto u_row_ptrs = u_row_ptrs_array.get_data();
    exec->run(make_initialize_row_ptrs_l_u(csr_system_matrix, l_row_ptrs,
                                           u_row_ptrs));

    auto l_nnz =
        static_cast<size_type>(exec->copy_val_to_host(l_row_ptrs + num_rows));
    auto u_nnz =
        static_cast<size_type>(exec->copy_val_to_host(u_row_ptrs + num_rows));

    auto mtx_size = csr_system_matrix->get_size();
    auto l = CsrMatrix::create(exec, mtx_size, Array<ValueType>{exec, l_nnz},
                               Array<IndexType>{exec, l_nnz},
                               std::move(l_row_ptrs_array));
    auto u = CsrMatrix::create(exec, mtx_size, Array<ValueType>{exec, u_nnz},
                               Array<IndexType>{exec, u_nnz},
                               std::move(u_row_ptrs_array));

    // initialize L and U
    exec->run(make_initialize_l_u(csr_system_matrix, l.get(), u.get()));

    // compute limit #nnz for L and U
    auto l_nnz_limit =
        static_cast<IndexType>(l_nnz * parameters_.fill_in_limit);
    auto u_nnz_limit =
        static_cast<IndexType>(u_nnz * parameters_.fill_in_limit);

    ParIlutState<ValueType, IndexType> state{exec,
                                             csr_system_matrix,
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
void ParIlutState<ValueType, IndexType>::iterate()
{
    // compute L * U
    exec->run(make_spgemm(l.get(), u.get(), lu.get()));

    // add new candidates to L' and U' factors
    exec->run(make_add_candidates(lu.get(), system_matrix, l.get(), u.get(),
                                  l_new.get(), u_new.get()));

    // update U'(CSC), L'(COO), U'(COO) sizes and pointers
    {
        auto l_nnz = l_new->get_num_stored_elements();
        auto u_nnz = u_new->get_num_stored_elements();
        CooBuilder l_builder{l_coo.get()};
        CooBuilder u_builder{u_coo.get()};
        CsrBuilder u_csc_builder{u_new_csc.get()};
        // resize arrays that will be filled
        l_builder.get_row_idx_array().resize_and_reset(l_nnz);
        u_builder.get_row_idx_array().resize_and_reset(u_nnz);
        u_csc_builder.get_col_idx_array().resize_and_reset(u_nnz);
        u_csc_builder.get_value_array().resize_and_reset(u_nnz);
        // update arrays that will be aliased
        l_builder.get_col_idx_array() =
            Array<IndexType>::view(exec, l_nnz, l_new->get_col_idxs());
        u_builder.get_col_idx_array() =
            Array<IndexType>::view(exec, u_nnz, u_new->get_col_idxs());
        l_builder.get_value_array() =
            Array<ValueType>::view(exec, l_nnz, l_new->get_values());
        u_builder.get_value_array() =
            Array<ValueType>::view(exec, u_nnz, u_new->get_values());
    }

    // convert U' into CSC format
    exec->run(make_csr_transpose(u_new.get(), u_new_csc.get()));

    // convert L' and U' into COO format
    exec->run(make_convert_to_coo(l_new.get(), l_coo.get()));
    exec->run(make_convert_to_coo(u_new.get(), u_coo.get()));

    // execute asynchronous iteration
    exec->run(make_compute_l_u_factors(system_matrix, l_new.get(), l_coo.get(),
                                       u_new.get(), u_coo.get(),
                                       u_new_csc.get()));

    // determine ranks for selection/filtering
    IndexType l_nnz = l_new->get_num_stored_elements();
    IndexType u_nnz = u_new->get_num_stored_elements();
    // make sure that the rank is in [0, *_nnz)
    auto l_filter_rank = std::max<IndexType>(0, l_nnz - l_nnz_limit - 1);
    auto u_filter_rank = std::max<IndexType>(0, u_nnz - u_nnz_limit - 1);
    remove_complex<ValueType> l_threshold{};
    remove_complex<ValueType> u_threshold{};
    CooMatrix *null_coo = nullptr;
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

    // execute asynchronous iteration
    exec->run(make_compute_l_u_factors(system_matrix, l.get(), l_coo.get(),
                                       u.get(), u_coo.get(), u_csc.get()));
}


#define GKO_DECLARE_PAR_ILUT(ValueType, IndexType) \
    class ParIlut<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PAR_ILUT);


}  // namespace factorization
}  // namespace gko