// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/preconditioner/ilu.hpp"

#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/config/type_descriptor.hpp>

#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"

namespace gko {
namespace preconditioner {


template <typename ValueType, bool ReverseApply, typename IndexType>
typename Ilu<ValueType, ReverseApply, IndexType>::parameters_type&
Ilu<ValueType, ReverseApply, IndexType>::parameters_type::with_l_solver(
    deferred_factory_parameter<const LinOpFactory> solver)

{
    this->l_solver_generator = std::move(solver);
    this->deferred_factories["l_solver"] = [](const auto& exec, auto& params) {
        if (!params.l_solver_generator.is_empty()) {
            params.l_solver_factory = params.l_solver_generator.on(exec);
        }
    };
    return *this;
}


template <typename ValueType, bool ReverseApply, typename IndexType>
typename Ilu<ValueType, ReverseApply, IndexType>::parameters_type
Ilu<ValueType, ReverseApply, IndexType>::parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    auto params = Ilu::build();
    // reverse_apply is used for determining the Ilu type
    config::config_check_decorator config_check(config, {"reverse_apply"});

    if (auto& obj = config_check.get("l_solver")) {
        params.with_l_solver(
            gko::config::parse_or_get_factory<const LinOpFactory>(
                obj, context, td_for_child));
    }
    if (auto& obj = config_check.get("u_solver")) {
        params.with_u_solver(
            gko::config::parse_or_get_factory<const LinOpFactory>(
                obj, context, td_for_child));
    }
    if (auto& obj = config_check.get("factorization")) {
        params.with_factorization(
            config::parse_or_get_factory<const LinOpFactory>(obj, context,
                                                             td_for_child));
    }

    return params;
}


template <typename ValueType, bool ReverseApply, typename IndexType>
std::unique_ptr<LinOp> Ilu<ValueType, ReverseApply, IndexType>::transpose()
    const

{
    std::unique_ptr<transposed_type> transposed{
        new transposed_type{this->get_executor()}};
    transposed->set_size(gko::transpose(this->get_size()));
    transposed->l_solver_ =
        share(as<Transposable>(this->get_u_solver())->transpose());
    transposed->u_solver_ =
        share(as<Transposable>(this->get_l_solver())->transpose());

    return std::move(transposed);
}


template <typename ValueType, bool ReverseApply, typename IndexType>
std::unique_ptr<LinOp> Ilu<ValueType, ReverseApply, IndexType>::conj_transpose()
    const

{
    std::unique_ptr<transposed_type> transposed{
        new transposed_type{this->get_executor()}};
    transposed->set_size(gko::transpose(this->get_size()));
    transposed->l_solver_ =
        share(as<Transposable>(this->get_u_solver())->conj_transpose());
    transposed->u_solver_ =
        share(as<Transposable>(this->get_l_solver())->conj_transpose());

    return std::move(transposed);
}


template <typename ValueType, bool ReverseApply, typename IndexType>
Ilu<ValueType, ReverseApply, IndexType>&
Ilu<ValueType, ReverseApply, IndexType>::operator=(const Ilu& other)
{
    if (&other != this) {
        LinOp::operator=(other);
        auto exec = this->get_executor();
        l_solver_ = other.l_solver_;
        u_solver_ = other.u_solver_;
        parameters_ = other.parameters_;
        if (other.get_executor() != exec) {
            l_solver_ = gko::clone(exec, l_solver_);
            u_solver_ = gko::clone(exec, u_solver_);
        }
    }
    return *this;
}


template <typename ValueType, bool ReverseApply, typename IndexType>
Ilu<ValueType, ReverseApply, IndexType>&
Ilu<ValueType, ReverseApply, IndexType>::operator=(Ilu&& other)

{
    if (&other != this) {
        LinOp::operator=(other);
        auto exec = this->get_executor();
        l_solver_ = std::move(other.l_solver_);
        u_solver_ = std::move(other.u_solver_);
        parameters_ = std::exchange(other.parameters_, parameters_type{});
        if (other.get_executor() != exec) {
            l_solver_ = gko::clone(exec, l_solver_);
            u_solver_ = gko::clone(exec, u_solver_);
        }
    }
    return *this;
}


template <typename ValueType, bool ReverseApply, typename IndexType>
Ilu<ValueType, ReverseApply, IndexType>::Ilu(const Ilu& other)
    : Ilu{other.get_executor()}
{
    *this = other;
}


template <typename ValueType, bool ReverseApply, typename IndexType>
Ilu<ValueType, ReverseApply, IndexType>::Ilu(Ilu&& other)
    : Ilu{other.get_executor()}
{
    *this = std::move(other);
}


template <typename ValueType, bool ReverseApply, typename IndexType>
void Ilu<ValueType, ReverseApply, IndexType>::apply_impl(const LinOp* b,
                                                         LinOp* x) const

{
    // take care of real-to-complex apply
    precision_dispatch_real_complex<value_type>(
        [&](auto dense_b, auto dense_x) {
            this->set_cache_to(dense_b);
            if (!ReverseApply) {
                l_solver_->apply(dense_b, cache_.intermediate);
                if (u_solver_->apply_uses_initial_guess()) {
                    dense_x->copy_from(
                        as<ClonableObject>(cache_.intermediate.get()));
                }
                u_solver_->apply(cache_.intermediate, dense_x);
            } else {
                u_solver_->apply(dense_b, cache_.intermediate);
                if (l_solver_->apply_uses_initial_guess()) {
                    dense_x->copy_from(
                        as<ClonableObject>(cache_.intermediate.get()));
                }
                l_solver_->apply(cache_.intermediate, dense_x);
            }
        },
        b, x);
}


template <typename ValueType, bool ReverseApply, typename IndexType>
void Ilu<ValueType, ReverseApply, IndexType>::apply_impl(const LinOp* alpha,
                                                         const LinOp* b,
                                                         const LinOp* beta,
                                                         LinOp* x) const
{
    precision_dispatch_real_complex<value_type>(
        [&](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            this->set_cache_to(dense_b);
            if (!ReverseApply) {
                l_solver_->apply(dense_b, cache_.intermediate);
                u_solver_->apply(dense_alpha, cache_.intermediate, dense_beta,
                                 dense_x);
            } else {
                u_solver_->apply(dense_b, cache_.intermediate);
                l_solver_->apply(dense_alpha, cache_.intermediate, dense_beta,
                                 dense_x);
            }
        },
        alpha, b, beta, x);
}


template <typename ValueType, bool ReverseApply, typename IndexType>
Ilu<ValueType, ReverseApply, IndexType>::Ilu(
    std::shared_ptr<const Executor> exec)
    : LinOp(std::move(exec))
{}


template <typename ValueType, bool ReverseApply, typename IndexType>
Ilu<ValueType, ReverseApply, IndexType>::Ilu(
    const Factory* factory, std::shared_ptr<const LinOp> lin_op)
    : LinOp(factory->get_executor(), lin_op->get_size()),
      parameters_{factory->get_parameters()}
{
    auto comp =
        std::dynamic_pointer_cast<const Composition<value_type>>(lin_op);
    std::shared_ptr<const LinOp> l_factor;
    std::shared_ptr<const LinOp> u_factor;

    // build factorization if we weren't passed a composition
    if (!comp) {
        auto exec = lin_op->get_executor();
        if (!parameters_.factorization_factory) {
            parameters_.factorization_factory =
                factorization::ParIlu<value_type, index_type>::build().on(exec);
        }
        auto fact = std::shared_ptr<const LinOp>(
            parameters_.factorization_factory->generate(lin_op));
        // ensure that the result is a composition
        comp = as<const Composition<value_type>>(fact);
    }
    if (comp->get_operators().size() == 2) {
        l_factor = comp->get_operators()[0];
        u_factor = comp->get_operators()[1];
    } else {
        GKO_NOT_SUPPORTED(comp);
    }
    GKO_ASSERT_EQUAL_DIMENSIONS(l_factor, u_factor);

    auto exec = this->get_executor();

    // If no factories are provided, generate default ones
    if (!parameters_.l_solver_factory) {
        // when not providing l_solver_factory, use LowerTrs as the default one
        l_solver_ = solver::LowerTrs<value_type, index_type>::build()
                        .on(exec)
                        ->generate(l_factor);
    } else {
        l_solver_ = parameters_.l_solver_factory->generate(l_factor);
    }
    if (!parameters_.u_solver_factory) {
        // when not providing u_solver_factory, use UpperTrs as the default one
        u_solver_ = solver::UpperTrs<value_type, index_type>::build()
                        .on(exec)
                        ->generate(u_factor);
    } else {
        u_solver_ = parameters_.u_solver_factory->generate(u_factor);
    }
}


template <typename ValueType, bool ReverseApply, typename IndexType>
void Ilu<ValueType, ReverseApply, IndexType>::set_cache_to(const LinOp* b) const

{
    if (cache_.intermediate == nullptr) {
        cache_.intermediate =
            matrix::Dense<value_type>::create(this->get_executor());
    }
    // Use b as the initial guess for the first triangular solve
    as<ClonableObject>(cache_.intermediate.get())
        ->copy_from(as<ClonableObject>(b));
}


// only instantiate the value type variants of ILU, whose solver is LinOp.
#define GKO_DECLARE_ILU_FALSE(ValueType, IndexType) \
    class Ilu<ValueType, false, IndexType>
#define GKO_DECLARE_ILU_TRUE(ValueType, IndexType) \
    class Ilu<ValueType, true, IndexType>

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ILU_FALSE);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ILU_TRUE);


}  // namespace preconditioner
}  // namespace gko
