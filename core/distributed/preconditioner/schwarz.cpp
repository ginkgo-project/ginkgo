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

#include "core/base/utils.hpp"
#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"
#include "core/distributed/helpers.hpp"


namespace gko {
namespace experimental {
namespace distributed {
namespace preconditioner {


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
typename Schwarz<ValueType, LocalIndexType, GlobalIndexType>::parameters_type
Schwarz<ValueType, LocalIndexType, GlobalIndexType>::parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    auto params = Schwarz<ValueType, LocalIndexType, GlobalIndexType>::build();

    if (auto& obj = config.get("generated_local_solver")) {
        params.with_generated_local_solver(
            gko::config::get_stored_obj<const LinOp>(obj, context));
    }
    if (auto& obj = config.get("local_solver")) {
        params.with_local_solver(
            gko::config::parse_or_get_factory<const LinOpFactory>(
                obj, context, td_for_child));
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

    if (this->local_solver_ != nullptr) {
        this->local_solver_->apply(gko::detail::get_local(dense_b),
                                   gko::detail::get_local(dense_x));
    }

    if (this->coarse_solver_ != nullptr && this->galerkin_ops_ != nullptr) {
        auto restrict = this->galerkin_ops_->get_restrict_op();
        auto prolong = this->galerkin_ops_->get_prolong_op();

        restrict->apply(dense_b, this->csol_);
        this->coarse_solver_->apply(this->csol_, this->csol_);
        prolong->apply(this->half_.get(), this->csol_.get(), this->half_.get(),
                       dense_x);
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
    auto dist_mat =
        as<experimental::distributed::Matrix<ValueType, LocalIndexType,
                                             GlobalIndexType>>(system_matrix);

    if (parameters_.local_solver) {
        this->set_solver(gko::share(
            parameters_.local_solver->generate(dist_mat->get_local_matrix())));
    } else {
        this->set_solver(parameters_.generated_local_solver);
    }


    if (parameters_.galerkin_ops_factory && parameters_.coarse_solver_factory) {
        this->galerkin_ops_ = as<multigrid::MultigridLevel>(
            share(parameters_.galerkin_ops_factory->generate(dist_mat)));
        auto coarse =
            as<experimental::distributed::Matrix<ValueType, LocalIndexType,
                                                 GlobalIndexType>>(
                this->galerkin_ops_->get_coarse_op());
        auto exec = coarse->get_executor();
        auto comm = coarse->get_communicator();
        this->coarse_solver_ =
            parameters_.coarse_solver_factory->generate(coarse);
        // TODO: Set correct rhs and stride.
        auto cs_ncols = 1;  // dense_x->get_size()[1];
        auto cs_local_nrows = coarse->get_local_matrix()->get_size()[0];
        auto cs_global_nrows = coarse->get_size()[0];
        auto cs_local_size = dim<2>(cs_local_nrows, cs_ncols);
        auto cs_global_size = dim<2>(cs_global_nrows, cs_ncols);
        this->csol_ = gko::share(dist_vec::create(exec, comm, cs_global_size,
                                                  cs_local_size,
                                                  1 /*dense_x->get_stride()*/));
        // this->temp_ = this->csol->clone();
        this->half_ = gko::share(gko::initialize<Vector>({0.5}, exec));
    }
}


#define GKO_DECLARE_SCHWARZ(ValueType, LocalIndexType, GlobalIndexType) \
    class Schwarz<ValueType, LocalIndexType, GlobalIndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(GKO_DECLARE_SCHWARZ);


}  // namespace preconditioner
}  // namespace distributed
}  // namespace experimental
}  // namespace gko
