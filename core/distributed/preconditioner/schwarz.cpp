// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/distributed/preconditioner/schwarz.hpp"

#include <memory>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/temporary_conversion.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/multigrid/multigrid_level.hpp>

#include "core/base/utils.hpp"
#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"
#include "core/distributed/helpers.hpp"
#include "core/matrix/csr_kernels.hpp"

namespace gko {
namespace experimental {
namespace distributed {
namespace preconditioner {
namespace {


GKO_REGISTER_OPERATION(row_wise_absolute_sum, csr::row_wise_absolute_sum);


}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
typename Schwarz<ValueType, LocalIndexType, GlobalIndexType>::parameters_type
Schwarz<ValueType, LocalIndexType, GlobalIndexType>::parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    auto params = Schwarz::build();
    config::config_check_decorator config_check(config);
    if (auto& obj = config_check.get("generated_local_solver")) {
        params.with_generated_local_solver(
            config::get_stored_obj<const LinOp>(obj, context));
    }
    if (auto& obj = config_check.get("local_solver")) {
        params.with_local_solver(
            config::parse_or_get_factory<const LinOpFactory>(obj, context,
                                                             td_for_child));
    }
    if (auto& obj = config_check.get("l1_smoother")) {
        params.with_l1_smoother(obj.get_boolean());
    }
    if (auto& obj = config_check.get("coarse_level")) {
        params.with_coarse_level(
            gko::config::parse_or_get_factory<const LinOpFactory>(
                obj, context, td_for_child));
    }
    if (auto& obj = config_check.get("coarse_solver")) {
        params.with_coarse_solver(
            gko::config::parse_or_get_factory<const LinOpFactory>(
                obj, context, td_for_child));
    }
    if (auto& obj = config_check.get("coarse_weight")) {
        params.with_coarse_weight(gko::config::get_value<ValueType>(obj));
    }

    return params;
}

template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
bool Schwarz<ValueType, LocalIndexType,
             GlobalIndexType>::apply_uses_initial_guess() const
{
    return this->local_solver_->apply_uses_initial_guess();
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
    using dist_vec = experimental::distributed::Vector<ValueType>;
    auto exec = this->get_executor();

    // Two-level
    if (this->coarse_solver_ != nullptr && this->coarse_level_ != nullptr) {
        if (this->local_solver_) {
            this->local_solver_->apply(gko::detail::get_local(dense_b),
                                       gko::detail::get_local(dense_x));
        }
        auto coarse_level =
            as<gko::multigrid::MultigridLevel>(this->coarse_level_);
        auto restrict_op = coarse_level->get_restrict_op();
        auto prolong_op = coarse_level->get_prolong_op();
        auto coarse_op =
            as<experimental::distributed::Matrix<ValueType, LocalIndexType,
                                                 GlobalIndexType>>(
                coarse_level->get_coarse_op());

        // Coarse solve vector cache init
        // Should allocate only in the first apply call if the number of rhs is
        // unchanged.
        auto cs_ncols = dense_x->get_size()[1];
        auto cs_local_nrows = coarse_op->get_local_matrix()->get_size()[0];
        auto cs_global_nrows = coarse_op->get_size()[0];
        auto cs_local_size = dim<2>(cs_local_nrows, cs_ncols);
        auto cs_global_size = dim<2>(cs_global_nrows, cs_ncols);
        auto comm = coarse_op->get_communicator();
        csol_cache_.init(exec, comm, cs_global_size, cs_local_size);
        crhs_cache_.init(exec, comm, cs_global_size, cs_local_size);

        // Additive apply of coarse correction
        restrict_op->apply(dense_b, crhs_cache_.get());
        // TODO: Does it make sense to restrict dense_x (to csol_cache) to
        // provide a good initial guess for the coarse solver ?
        if (this->coarse_solver_->apply_uses_initial_guess()) {
            csol_cache_->copy_from(crhs_cache_.get());
        }
        this->coarse_solver_->apply(crhs_cache_.get(), csol_cache_.get());
        prolong_op->apply(this->coarse_weight_, csol_cache_.get(),
                          this->local_weight_, dense_x);
    } else if (this->local_solver_ != nullptr) {
        this->local_solver_->apply(gko::detail::get_local(dense_b),
                                   gko::detail::get_local(dense_x));
    }
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Schwarz<ValueType, LocalIndexType, GlobalIndexType>::apply_impl(
    const LinOp* alpha, const LinOp* b, const LinOp* beta, LinOp* x) const
{
    // only dispatch distributed case
    experimental::distributed::precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            cache_.init_from(dense_x);
            cache_->copy_from(dense_x);
            this->apply_impl(dense_b, cache_.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, cache_.get());
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
    using Vector = matrix::Dense<ValueType>;
    using dist_vec = experimental::distributed::Vector<ValueType>;
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
    if ((parameters_.coarse_level && !parameters_.coarse_solver) ||
        (!parameters_.coarse_level && parameters_.coarse_solver)) {
        GKO_INVALID_STATE(
            "Requires both coarse solver and coarse level to be set.");
    }

    auto dist_mat =
        as<Matrix<ValueType, LocalIndexType, GlobalIndexType>>(system_matrix);
    auto local_matrix = dist_mat->get_local_matrix();

    if (parameters_.l1_smoother) {
        auto exec = this->get_executor();

        using Csr = matrix::Csr<ValueType, LocalIndexType>;
        auto local_matrix_copy = share(Csr::create(exec));
        as<ConvertibleTo<Csr>>(local_matrix)->convert_to(local_matrix_copy);

        auto non_local_matrix = copy_and_convert_to<Csr>(
            exec, as<Matrix<ValueType, LocalIndexType, GlobalIndexType>>(
                      system_matrix)
                      ->get_non_local_matrix());

        array<ValueType> l1_diag_arr{exec, local_matrix->get_size()[0]};

        exec->run(
            make_row_wise_absolute_sum(non_local_matrix.get(), l1_diag_arr));

        // compute local_matrix_copy <- diag(l1) + local_matrix_copy
        auto l1_diag = matrix::Diagonal<ValueType>::create(
            exec, local_matrix->get_size()[0], std::move(l1_diag_arr));
        auto l1_diag_csr = Csr::create(exec);
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

    gko::remove_complex<ValueType> cweight =
        gko::detail::real_impl(parameters_.coarse_weight);
    if (cweight >= 0.0 && cweight <= 1.0) {
        this->local_weight_ = gko::initialize<matrix::Dense<ValueType>>(
            {one<ValueType>() -
             static_cast<ValueType>(parameters_.coarse_weight)},
            this->get_executor());
        this->coarse_weight_ = gko::initialize<matrix::Dense<ValueType>>(
            {static_cast<ValueType>(parameters_.coarse_weight)},
            this->get_executor());
    } else {
        this->local_weight_ = gko::initialize<matrix::Dense<ValueType>>(
            {one<ValueType>()}, this->get_executor());
        this->coarse_weight_ = gko::initialize<matrix::Dense<ValueType>>(
            {one<ValueType>()}, this->get_executor());
    }

    if (parameters_.coarse_level && parameters_.coarse_solver) {
        this->coarse_level_ =
            share(parameters_.coarse_level->generate(system_matrix));
        if (this->coarse_level_ == nullptr) {
            GKO_NOT_SUPPORTED(this->coarse_level_);
        }
        if (auto coarse = as<multigrid::MultigridLevel>(this->coarse_level_)
                              ->get_coarse_op()) {
            this->coarse_solver_ = share(parameters_.coarse_solver->generate(
                as<Matrix<ValueType, LocalIndexType, GlobalIndexType>>(
                    coarse)));
            if (this->coarse_solver_ == nullptr) {
                GKO_NOT_SUPPORTED(this->coarse_solver_);
            }
        }
    }
}


#define GKO_DECLARE_SCHWARZ(ValueType, LocalIndexType, GlobalIndexType) \
    class Schwarz<ValueType, LocalIndexType, GlobalIndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(GKO_DECLARE_SCHWARZ);


}  // namespace preconditioner
}  // namespace distributed
}  // namespace experimental
}  // namespace gko
