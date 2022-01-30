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

#include <ginkgo/core/preconditioner/schwarz.hpp>


#include <memory>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/temporary_conversion.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/utils.hpp"
#include "core/preconditioner/schwarz_kernels.hpp"


namespace gko {
namespace preconditioner {
namespace schwarz {
namespace {


GKO_REGISTER_OPERATION(apply, schwarz::apply);
// GKO_REGISTER_OPERATION(advanced_apply, schwarz::advanced_apply);


}  // anonymous namespace
}  // namespace schwarz


template <typename ValueType, typename IndexType>
void Schwarz<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x);
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void Schwarz<ValueType, IndexType>::apply_dense_impl(
    const matrix::Dense<ValueType>* dense_b,
    matrix::Dense<ValueType>* dense_x) const
{
    using Vector = matrix::Dense<ValueType>;
    auto exec = this->get_executor();
    auto one_op = initialize<Vector>({one<ValueType>()}, exec);
    auto neg_one_op = initialize<Vector>({-one<ValueType>()}, exec);
    size_type offset = 0;

    for (size_type i = 0; i < this->num_subdomains_; ++i) {
        size_type l_num_rows = this->subdomain_matrices_[i]->get_size()[0];
        auto rspan = gko::span(offset, offset + l_num_rows);
        const auto b_view = dense_b->create_submatrix(
            rspan, gko::span(0, dense_b->get_size()[1]));
        auto x_view = dense_x->create_submatrix(
            rspan, gko::span(0, dense_x->get_size()[1]));
        this->subdomain_solvers_[i]->apply(b_view.get(), x_view.get());
        offset += l_num_rows;
    }
    if (parameters_.coarse_solver.uses_coarse_solver_) {
        auto r = Vector::create_with_config_of(dense_b);
        r->copy_from(dense_b);
        auto restrict_op =
            parameters_.coarse_solver.coarse_op_->get_restrict_op();
        auto prolong_op =
            parameters_.coarse_solver.coarse_op_->get_prolong_op();

        system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(), r.get());
        auto restricted_r = Vector::create(
            exec,
            gko::dim<2>(restrict_op->get_size()[0], dense_x->get_size()[1]));
        restrict_op->apply(r.get(), restricted_r.get());
        auto restricted_e = Vector::create(
            exec,
            gko::dim<2>(restrict_op->get_size()[0], dense_x->get_size()[1]));
        restricted_e->fill(zero<ValueType>());
        parameters_.coarse_solver.coarse_solver_->apply(restricted_r.get(),
                                                        restricted_e.get());
        prolong_op->apply(one_op.get(), restricted_e.get(), one_op.get(),
                          dense_x);
    }
}


template <typename ValueType, typename IndexType>
void Schwarz<ValueType, IndexType>::apply_impl(const LinOp* alpha,
                                               const LinOp* b,
                                               const LinOp* beta,
                                               LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, const auto dense_b, auto dense_beta,
               auto dense_x) {
            size_type offset = 0;
            // TODO Replace with BlockApprox
            for (size_type i = 0; i < this->num_subdomains_; ++i) {
                size_type l_num_rows = parameters_.subdomain_sizes[i];
                auto rspan = gko::span(offset, offset + l_num_rows);
                const auto b_view = dense_b->create_submatrix(
                    rspan, gko::span(0, dense_b->get_size()[1]));
                auto x_view = dense_x->create_submatrix(
                    rspan, gko::span(0, dense_x->get_size()[1]));
                this->subdomain_solvers_[i]->apply(dense_alpha, b_view.get(),
                                                   dense_beta, x_view.get());
                offset += l_num_rows;
            }
        },
        alpha, b, beta, x);
}


template <typename ValueType, typename IndexType>
void Schwarz<ValueType, IndexType>::write(mat_data& data) const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Schwarz<ValueType, IndexType>::transpose() const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Schwarz<ValueType, IndexType>::conj_transpose() const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void Schwarz<ValueType, IndexType>::generate(const LinOp* system_matrix,
                                             bool skip_sorting)
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);
    using csr_type = matrix::Csr<ValueType, IndexType>;
    const auto exec = this->get_executor();
    auto csr_mtx =
        convert_to_with_sorting<csr_type>(exec, system_matrix, skip_sorting);
    size_type offset = 0;
    auto mat_size = system_matrix->get_size();
    std::vector<size_type> subd_sizes;
    if (parameters_.subdomain_sizes.size() == 0) {
        subd_sizes = std::vector<size_type>(num_subdomains_,
                                            mat_size[0] / num_subdomains_);
        subd_sizes[0] += mat_size[0] - (num_subdomains_ * subd_sizes[1]);
    } else {
        subd_sizes = parameters_.subdomain_sizes;
    }
    size_type l_num_rows = 0;
    // TODO Replace with BlockApprox
    for (size_type i = 0; i < num_subdomains_; ++i) {
        size_type l_num_rows = subd_sizes[i];
        auto rspan = gko::span(offset, offset + l_num_rows);
        auto cspan = gko::span(offset, offset + l_num_rows);
        subdomain_matrices_.emplace_back(
            gko::share(csr_mtx->create_submatrix(rspan, cspan)));
        offset += l_num_rows;
    }
    if (parameters_.generated_inner_solvers.size() > 0) {
        GKO_ASSERT(parameters_.generated_inner_solvers.size() ==
                   num_subdomains_);
    } else if (parameters_.inner_solver) {
        for (size_type i = 0; i < num_subdomains_; ++i) {
            subdomain_solvers_.emplace_back(
                parameters_.inner_solver->generate(subdomain_matrices_[i]));
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
    if (parameters_.coarse_solver.uses_coarse_solver_) {
        coarse_solver_ = parameters_.coarse_solver.coarse_solver_;
    }
}


#define GKO_DECLARE_SCHWARZ(ValueType, IndexType) \
    class Schwarz<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SCHWARZ);


}  // namespace preconditioner
}  // namespace gko
