/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/preconditioner/ras.hpp>


#include <memory>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/distributed/block_approx.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/matrix/block_approx.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/extended_float.hpp"
#include "core/base/utils.hpp"
#include "core/preconditioner/distributed_helpers.hpp"
#include "core/preconditioner/ras_kernels.hpp"


namespace gko {
namespace preconditioner {
namespace ras {


GKO_REGISTER_OPERATION(simple_apply, ras::simple_apply);
GKO_REGISTER_OPERATION(apply, ras::apply);
GKO_REGISTER_OPERATION(find_blocks, ras::find_blocks);
GKO_REGISTER_OPERATION(generate, ras::generate);
GKO_REGISTER_OPERATION(transpose_ras, ras::transpose_ras);
GKO_REGISTER_OPERATION(conj_transpose_ras, ras::conj_transpose_ras);
GKO_REGISTER_OPERATION(convert_to_dense, ras::convert_to_dense);
GKO_REGISTER_OPERATION(initialize_precisions, ras::initialize_precisions);


}  // namespace ras


template <typename ValueType, typename IndexType>
void Ras<ValueType, IndexType>::apply_impl(const LinOp *b, LinOp *x) const
{
    precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x);
        },
        b, x);
}


template <typename ValueType, typename IndexType>
template <typename VectorType>
void Ras<ValueType, IndexType>::apply_dense_impl(const VectorType *dense_b,
                                                 VectorType *dense_x) const
{
    using LocalVector = matrix::Dense<ValueType>;
    if (is_distributed()) {
        GKO_ASSERT(this->inner_solvers_.size() > 0);
        this->inner_solvers_[0]->apply(detail::get_local(dense_b),
                                       detail::get_local(dense_x));
    } else {
        using block_t = matrix::BlockApprox<matrix::Csr<ValueType, IndexType>>;
        auto block_mtx = as<block_t>(block_system_matrix_);
        auto block_ptrs_arr = block_mtx->get_block_ptrs_array();
        block_ptrs_arr.set_executor(this->get_executor()->get_master());
        auto block_ptrs = block_ptrs_arr.get_const_data();
        auto num_subdomains = this->inner_solvers_.size();
        auto non_ov_bsize = block_mtx->get_non_overlap_block_sizes();
        auto left_ov = block_mtx->get_block_left_overlaps();
        auto right_ov = block_mtx->get_block_right_overlaps();
        for (size_type i = 0; i < num_subdomains; ++i) {
            auto block_size = this->inner_solvers_[i]->get_size();
            auto b_view =
                detail::create_submatrix(dense_b,
                                         span{block_ptrs[i] - left_ov[i],
                                              block_ptrs[i + 1] + right_ov[i]},
                                         span{0, dense_x->get_size()[1]});
            auto x_view =
                detail::create_submatrix(dense_x,
                                         span{block_ptrs[i] - left_ov[i],
                                              block_ptrs[i + 1] + right_ov[i]},
                                         span{0, dense_x->get_size()[1]});
            if (block_mtx->has_overlap()) {
                this->inner_solvers_[i]->apply(
                    b_view.get(), x_view.get(),
                    gko::OverlapMask{
                        gko::span{
                            left_ov[i],
                            left_ov[i] + non_ov_bsize.get_const_data()[i]},
                        true});
            } else {
                this->inner_solvers_[i]->apply(b_view.get(), x_view.get());
            }
        }
    }
    if (this->coarse_solvers_.size() > 0) {
        auto corr = dense_x->clone();
        auto res = dense_b->clone();
        auto one = initialize<LocalVector>({1.0}, this->get_executor());
        auto neg_one = initialize<LocalVector>({-1.0}, this->get_executor());
        this->system_matrix_->apply(lend(neg_one), dense_x, lend(one),
                                    res.get());
        for (size_type i = 0; i < this->coarse_solvers_.size(); ++i) {
            auto rel_fac = parameters_.coarse_relaxation_factors[0];
            if (parameters_.coarse_relaxation_factors.size() > 1) {
                rel_fac = parameters_.coarse_relaxation_factors[i];
            }
            this->coarse_solvers_[i]->apply(res.get(), corr.get());
        }
        auto fac = initialize<LocalVector>({1.0}, this->get_executor());
        dense_x->add_scaled(fac.get(), corr.get());
    }
}


template <typename ValueType, typename IndexType>
void Ras<ValueType, IndexType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                           const LinOp *beta, LinOp *x) const
{
    precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_dense_impl(dense_b, x_clone.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, x_clone.get());
        },
        alpha, b, beta, x);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Ras<ValueType, IndexType>::transpose() const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Ras<ValueType, IndexType>::conj_transpose() const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void Ras<ValueType, IndexType>::generate(const LinOp *system_matrix)
{
    using base_mat = matrix::Csr<ValueType, IndexType>;
    using dist_mat = distributed::Matrix<ValueType, IndexType>;
    using block_t = matrix::BlockApprox<base_mat>;
    using dist_block_t = distributed::BlockApprox<ValueType, IndexType>;
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);
    if (dynamic_cast<const base_mat *>(system_matrix) != nullptr) {
        auto mat = as<const base_mat>(system_matrix);
        this->block_system_matrix_ =
            block_t::create(mat->get_executor(), mat,
                            parameters_.block_dimensions, this->overlaps_);
        auto block_mtxs =
            as<block_t>(this->block_system_matrix_)->get_block_mtxs();
        this->block_dims_ =
            as<block_t>(this->block_system_matrix_)->get_block_dimensions();
        const auto num_subdomains = block_mtxs.size();
        for (size_type i = 0; i < num_subdomains; ++i) {
            this->inner_solvers_.emplace_back(
                parameters_.inner_solver->generate(block_mtxs[i]));
        }
        if (parameters_.coarse_solvers[0]) {
            for (size_type i = 0; i < parameters_.coarse_solvers.size(); ++i) {
                this->coarse_solvers_.emplace_back(
                    parameters_.coarse_solvers[i]->generate(
                        this->system_matrix_));
            }
        }
        this->is_distributed_ = false;
    } else if (dynamic_cast<const block_t *>(system_matrix) != nullptr) {
        this->block_system_matrix_ = this->system_matrix_;
        auto block_mtxs = as<block_t>(system_matrix)->get_block_mtxs();
        this->overlaps_ = as<block_t>(system_matrix)->get_overlaps();
        this->block_dims_ = as<block_t>(system_matrix)->get_block_dimensions();
        const auto num_subdomains = block_mtxs.size();
        for (size_type i = 0; i < num_subdomains; ++i) {
            this->inner_solvers_.emplace_back(
                parameters_.inner_solver->generate(block_mtxs[i]));
        }
        this->is_distributed_ = false;
    } else if (dynamic_cast<const dist_mat *>(system_matrix) != nullptr) {
        auto mat = as<const dist_mat>(system_matrix);
        if (mat->get_executor() != this->get_executor()) {
            GKO_NOT_IMPLEMENTED;
        }

        if (parameters_.inner_solver) {
            this->block_system_matrix_ = dist_block_t::create(
                mat->get_executor(), mat, mat->get_communicator());
            auto block_mtxs =
                as<dist_block_t>(this->block_system_matrix_)->get_block_mtxs();
            const auto num_subdomains = block_mtxs.size();
            for (size_type i = 0; i < num_subdomains; ++i) {
                this->inner_solvers_.emplace_back(
                    parameters_.inner_solver->generate(
                        gko::share(block_mtxs[i])));
            }
        }
        if (this->inner_solvers_.size() < 1) {
            GKO_NOT_IMPLEMENTED;
        }
        if (parameters_.coarse_solvers[0]) {
            for (size_type i = 0; i < parameters_.coarse_solvers.size(); ++i) {
                this->coarse_solvers_.emplace_back(
                    parameters_.coarse_solvers[i]->generate(
                        this->system_matrix_));
            }
        }
        this->is_distributed_ = true;
    } else if (dynamic_cast<const dist_block_t *>(system_matrix) != nullptr) {
        this->block_system_matrix_ = this->system_matrix_;
        auto block_mtxs = as<dist_block_t>(system_matrix)->get_block_mtxs();
        const auto num_subdomains = block_mtxs.size();
        for (size_type i = 0; i < num_subdomains; ++i) {
            this->inner_solvers_.emplace_back(
                parameters_.inner_solver->generate(gko::share(block_mtxs[i])));
        }
    }
}


#define GKO_DECLARE_RAS(ValueType, IndexType) class Ras<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_RAS);


}  // namespace preconditioner
}  // namespace gko
