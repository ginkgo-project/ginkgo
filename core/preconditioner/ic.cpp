// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/preconditioner/ic.hpp"

#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils_helper.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>

#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"


namespace gko {
namespace preconditioner {


template <typename ValueType, typename IndexType>
typename Ic<ValueType, IndexType>::parameters_type
Ic<ValueType, IndexType>::parse(const config::pnode& config,
                                const config::registry& context,
                                const config::type_descriptor& td_for_child)

{
    auto params = Ic::build();
    config::config_check_decorator config_check(config);

    if (auto& obj = config_check.get("l_solver")) {
        params.with_l_solver(
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


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Ic<ValueType, IndexType>::transpose() const

{
    std::unique_ptr<transposed_type> transposed{
        new transposed_type{this->get_executor()}};
    transposed->set_size(gko::transpose(this->get_size()));
    transposed->l_solver_ =
        share(as<Transposable>(this->get_lh_solver())->transpose());
    transposed->lh_solver_ =
        share(as<Transposable>(this->get_l_solver())->transpose());

    return std::move(transposed);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Ic<ValueType, IndexType>::conj_transpose() const

{
    std::unique_ptr<transposed_type> transposed{
        new transposed_type{this->get_executor()}};
    transposed->set_size(gko::transpose(this->get_size()));
    transposed->l_solver_ =
        share(as<Transposable>(this->get_lh_solver())->conj_transpose());
    transposed->lh_solver_ =
        share(as<Transposable>(this->get_l_solver())->conj_transpose());

    return std::move(transposed);
}


template <typename ValueType, typename IndexType>
Ic<ValueType, IndexType>& Ic<ValueType, IndexType>::operator=(const Ic& other)
{
    if (&other != this) {
        LinOp::operator=(other);
        auto exec = this->get_executor();
        l_solver_ = other.l_solver_;
        lh_solver_ = other.lh_solver_;
        parameters_ = other.parameters_;
        if (other.get_executor() != exec) {
            l_solver_ = gko::clone(exec, l_solver_);
            lh_solver_ = gko::clone(exec, lh_solver_);
        }
    }
    return *this;
}


template <typename ValueType, typename IndexType>
Ic<ValueType, IndexType>& Ic<ValueType, IndexType>::operator=(Ic&& other)
{
    if (&other != this) {
        LinOp::operator=(other);
        auto exec = this->get_executor();
        l_solver_ = std::move(other.l_solver_);
        lh_solver_ = std::move(other.lh_solver_);
        parameters_ = std::exchange(other.parameters_, parameters_type{});
        if (other.get_executor() != exec) {
            l_solver_ = gko::clone(exec, l_solver_);
            lh_solver_ = gko::clone(exec, lh_solver_);
        }
    }
    return *this;
}


template <typename ValueType, typename IndexType>
Ic<ValueType, IndexType>::Ic(const Ic& other) : Ic{other.get_executor()}
{
    *this = other;
}


template <typename ValueType, typename IndexType>
Ic<ValueType, IndexType>::Ic(Ic&& other) : Ic{other.get_executor()}
{
    *this = std::move(other);
}


template <typename ValueType, typename IndexType>
void Ic<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const

{
    // take care of real-to-complex apply
    precision_dispatch_real_complex<value_type>(
        [&](auto dense_b, auto dense_x) {
            this->set_cache_to(dense_b);
            l_solver_->apply(dense_b, cache_.intermediate);
            if (lh_solver_->apply_uses_initial_guess()) {
                dense_x->copy_from(as<Cloneable>(cache_.intermediate.get()));
            }
            lh_solver_->apply(cache_.intermediate, dense_x);
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void Ic<ValueType, IndexType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                          const LinOp* beta, LinOp* x) const

{
    precision_dispatch_real_complex<value_type>(
        [&](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            this->set_cache_to(dense_b);
            l_solver_->apply(dense_b, cache_.intermediate);
            lh_solver_->apply(dense_alpha, cache_.intermediate, dense_beta,
                              dense_x);
        },
        alpha, b, beta, x);
}


template <typename ValueType, typename IndexType>
Ic<ValueType, IndexType>::Ic(std::shared_ptr<const Executor> exec)
    : LinOp(std::move(exec))
{}


template <typename ValueType, typename IndexType>
Ic<ValueType, IndexType>::Ic(const Factory* factory,
                             std::shared_ptr<const LinOp> lin_op)
    : LinOp(factory->get_executor(), lin_op->get_size()),
      parameters_{factory->get_parameters()}
{
    auto comp =
        std::dynamic_pointer_cast<const Composition<value_type>>(lin_op);
    std::shared_ptr<const LinOp> l_factor;

    // build factorization if we weren't passed a composition
    if (!comp) {
        auto exec = lin_op->get_executor();

        if (!parameters_.factorization_factory) {
            parameters_.factorization_factory =
                factorization::ParIc<value_type, index_type>::build()
                    .with_both_factors(false)
                    .on(exec);
        }
        auto fact = std::shared_ptr<const LinOp>(
            parameters_.factorization_factory->generate(lin_op));
        // ensure that the result is a composition
        comp = gko::as<const Composition<value_type>>(fact);
    }
    // comp must contain one or two factors
    if (comp->get_operators().size() > 2 || comp->get_operators().empty()) {
        GKO_NOT_SUPPORTED(comp);
    }
    l_factor = comp->get_operators()[0];
    GKO_ASSERT_IS_SQUARE_MATRIX(l_factor);

    auto exec = this->get_executor();

    // If no factories are provided, generate default ones
    if (!parameters_.l_solver_factory) {
        // when not providing l_solver_factory, use LowerTrs as the default one
        l_solver_ = solver::LowerTrs<value_type, index_type>::build()
                        .on(exec)
                        ->generate(l_factor);
        // If comp contains both factors: We only check the dimension from
        // the second factor. However, we still use the l_solver^H not
        // generate the solver on L^H to preserve the Hermitian property of
        // this preconditioner. LSolver(L)^H is not always LSolver^H(L^H).
        if (comp->get_operators().size() == 2) {
            auto lh_factor = comp->get_operators()[1];
            GKO_ASSERT_EQUAL_DIMENSIONS(l_factor, lh_factor);
        }
        lh_solver_ = as<Transposable>(l_solver_)->conj_transpose();
    } else {
        l_solver_ = parameters_.l_solver_factory->generate(l_factor);
        lh_solver_ = as<Transposable>(l_solver_)->conj_transpose();
    }
}


template <typename ValueType, typename IndexType>
void Ic<ValueType, IndexType>::set_cache_to(const LinOp* b) const

{
    if (cache_.intermediate == nullptr) {
        cache_.intermediate =
            matrix::MultiVector<value_type>::create(this->get_executor());
    }
    // Use b as the initial guess for the first triangular solve
    as<Cloneable>(cache_.intermediate.get())->copy_from(as<Cloneable>(b));
}


// only instantiate the value type variants of IC, whose solver is LinOp.
#define GKO_DECLARE_IC(ValueType, IndexType) class Ic<ValueType, IndexType>

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_IC);


}  // namespace preconditioner
}  // namespace gko
