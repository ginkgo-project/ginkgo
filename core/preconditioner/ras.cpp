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
        const auto dense_x_clone = (dense_x->clone());
        const auto num_subdomains = this->inner_solvers_.size();
        // FIXME host transfer
        auto temp_overlaps = this->overlaps_;
        temp_overlaps.set_executor(this->get_executor()->get_master());
        const auto block_overlaps = temp_overlaps.get_overlaps();
        const auto overlap_unidir = temp_overlaps.get_unidirectional_array();
        const auto overlap_start = temp_overlaps.get_overlap_at_start_array();
        size_type offset = 0;
        auto temp_x = detail::create_with_same_size(dense_x);
        for (size_type i = 0; i < num_subdomains; ++i) {
            auto overlap_start_offset =
                (block_overlaps && (!overlap_unidir[i] || overlap_start[i]))
                    ? block_overlaps[i]
                    : 0;
            auto overlap_end_offset =
                (block_overlaps && (!overlap_unidir[i] || !overlap_start[i]))
                    ? block_overlaps[i]
                    : 0;
            auto loc_size = this->block_dims_[i] - overlap_start_offset -
                            overlap_end_offset;
            auto row_span = span{offset - overlap_start_offset,
                                 offset + loc_size[0] + overlap_end_offset};
            auto col_span = span{0, dense_b->get_size()[1]};
            const auto loc_b = std::move(
                detail::create_submatrix(dense_b, row_span, col_span));
            auto x_row_span = span{offset, offset + loc_size[0]};
            auto x_col_span = span{0, dense_x->get_size()[1]};
            temp_x->copy_from(dense_x_clone.get());
            auto ov_x = std::move(
                detail::create_submatrix(temp_x.get(), row_span, col_span));
            this->inner_solvers_[i]->apply(loc_b.get(), ov_x.get());
            auto loc_x = std::move(
                detail::create_submatrix(dense_x, x_row_span, x_col_span));
            auto sol_view = std::move(
                detail::create_submatrix(temp_x.get(), x_row_span, x_col_span));
            loc_x->copy_from(sol_view.get());
            offset += loc_size[0];
        }
    }
}


template <typename ValueType, typename IndexType>
void Ras<ValueType, IndexType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                           const LinOp *beta, LinOp *x) const
{
    if (!is_distributed()) {
        using Dense = matrix::Dense<ValueType>;
        auto dense_b = const_cast<Dense *>(as<Dense>(b));
        auto dense_x = as<Dense>(x);
        const auto num_subdomains = this->inner_solvers_.size();
        size_type offset = 0;
        for (size_type i = 0; i < num_subdomains; ++i) {
            auto loc_size = this->inner_solvers_[i]->get_size();
            const auto loc_b = dense_b->create_submatrix(
                span{offset, offset + loc_size[0]}, span{0, b->get_size()[1]});
            auto loc_x = dense_x->create_submatrix(
                span{offset, offset + loc_size[0]}, span{0, x->get_size()[1]});
            this->inner_solvers_[i]->apply(alpha, loc_b.get(), beta,
                                           loc_x.get());
            offset += loc_size[0];
        }
    } else {
        using Dense = matrix::Dense<ValueType>;
        auto dense_b = const_cast<Dense *>(as<Dense>(b));
        auto dense_x = as<Dense>(x);
        const auto num_subdomains = this->inner_solvers_.size();
        size_type offset = 0;
        for (size_type i = 0; i < num_subdomains; ++i) {
            auto loc_size = this->inner_solvers_[i]->get_size();
            const auto loc_b = dense_b->create_submatrix(
                span{offset, offset + loc_size[0]}, span{0, b->get_size()[1]});
            auto loc_x = dense_x->create_submatrix(
                span{offset, offset + loc_size[0]}, span{0, x->get_size()[1]});
            this->inner_solvers_[i]->apply(alpha, loc_b.get(), beta,
                                           loc_x.get());
            offset += loc_size[0];
        }
    }
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
        this->is_distributed_ = false;
    } else if (dynamic_cast<const block_t *>(system_matrix) != nullptr) {
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
        this->block_system_matrix_ =
            dist_block_t::create(mat->get_executor(), mat);
        auto block_mtxs =
            as<dist_block_t>(this->block_system_matrix_)->get_block_mtxs();
        const auto num_subdomains = block_mtxs.size();
        for (size_type i = 0; i < num_subdomains; ++i) {
            this->inner_solvers_.emplace_back(
                parameters_.inner_solver->generate(gko::share(block_mtxs[i])));
        }
        this->is_distributed_ = true;
    } else if (dynamic_cast<const dist_block_t *>(system_matrix) != nullptr) {
        auto block_mtxs = as<dist_block_t>(system_matrix)->get_block_mtxs();
        const auto num_subdomains = block_mtxs.size();
        for (size_type i = 0; i < num_subdomains; ++i) {
            this->inner_solvers_.emplace_back(
                parameters_.inner_solver->generate(gko::share(block_mtxs[i])));
        }
        this->is_distributed_ = true;
    }
}


#define GKO_DECLARE_RAS(ValueType, IndexType) class Ras<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_RAS);


}  // namespace preconditioner
}  // namespace gko
