// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/distributed/preconditioner/schwarz.hpp>


#include <memory>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/temporary_conversion.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>


#include "core/base/utils.hpp"
#include "core/distributed/helpers.hpp"
#include "core/matrix/csr_kernels.hpp"


namespace gko {
namespace experimental {
namespace distributed {
namespace preconditioner {
namespace {


GKO_REGISTER_OPERATION(row_wise_sum, csr::row_wise_sum);


}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Schwarz<ValueType, LocalIndexType, GlobalIndexType>::apply_impl(
    const LinOp* b, LinOp* x) const
{
    precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x);
        },
        b, x);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
template <typename VectorType>
void Schwarz<ValueType, LocalIndexType, GlobalIndexType>::apply_dense_impl(
    const VectorType* dense_b, VectorType* dense_x) const
{
    using Vector = matrix::Dense<ValueType>;
    auto exec = this->get_executor();
    if (this->local_solver_ != nullptr) {
        this->local_solver_->apply(gko::detail::get_local(dense_b),
                                   gko::detail::get_local(dense_x));
    }
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Schwarz<ValueType, LocalIndexType, GlobalIndexType>::apply_impl(
    const LinOp* alpha, const LinOp* b, const LinOp* beta, LinOp* x) const
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


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Schwarz<ValueType, LocalIndexType, GlobalIndexType>::set_solver(
    std::shared_ptr<const LinOp> new_solver)
{
    auto exec = this->get_executor();
    if (new_solver) {
        if (new_solver->get_executor() != exec) {
            new_solver = gko::clone(exec, new_solver);
        }
    }
    this->local_solver_ = new_solver;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Schwarz<ValueType, LocalIndexType, GlobalIndexType>::generate(
    std::shared_ptr<const LinOp> system_matrix)
{
    if (parameters_.local_solver && parameters_.generated_local_solver) {
        GKO_INVALID_STATE(
            "Provided both a generated solver and a solver factory");
    }

    if (!parameters_.local_solver && !parameters_.generated_local_solver) {
        GKO_INVALID_STATE(
            "Requires either a generated solver or an solver factory");
    }

    if (parameters_.generated_local_solver) {
        this->set_solver(parameters_.generated_local_solver);
        return;
    }

    auto local_matrix =
        as<Matrix<ValueType, LocalIndexType, GlobalIndexType>>(system_matrix)
            ->get_local_matrix();

    if (parameters_.l1_smoother) {
        auto local_matrix_copy = share(clone(local_matrix));

        auto non_local_matrix = as<matrix::Csr<ValueType, LocalIndexType>>(
            as<Matrix<ValueType, LocalIndexType, GlobalIndexType>>(
                system_matrix)
                ->get_non_local_matrix());

        auto exec = this->get_executor();
        array<ValueType> l1_diag_arr{exec, local_matrix->get_size()[0]};

        exec->run(make_row_wise_sum(non_local_matrix.get(), l1_diag_arr, true));

        // compute local_matrix_copy <- diag(l1) + local_matrix_copy
        auto l1_diag = matrix::Diagonal<ValueType>::create(
            exec, local_matrix->get_size()[0], std::move(l1_diag_arr));
        auto l1_diag_csr = matrix::Csr<ValueType, LocalIndexType>::create(exec);
        l1_diag->move_to(l1_diag_csr);
        auto id = matrix::Identity<ValueType>::create(
            exec, local_matrix->get_size()[0]);
        auto one = initialize<matrix::Dense<ValueType>>(
            {::gko::one<ValueType>()}, exec);
        l1_diag_csr->apply(one, id, one, local_matrix_copy);

        this->set_solver(
            gko::share(parameters_.local_solver->generate(local_matrix_copy)));
    } else {
        this->set_solver(
            gko::share(parameters_.local_solver->generate(local_matrix)));
    }
}


#define GKO_DECLARE_SCHWARZ(ValueType, LocalIndexType, GlobalIndexType) \
    class Schwarz<ValueType, LocalIndexType, GlobalIndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(GKO_DECLARE_SCHWARZ);


}  // namespace preconditioner
}  // namespace distributed
}  // namespace experimental
}  // namespace gko
