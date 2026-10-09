// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/preconditioner/nullspace_projection.hpp"

#include <utility>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/matrix/identity.hpp>

#include "core/config/config_helper.hpp"
#include "core/distributed/helpers.hpp"


namespace gko {
namespace preconditioner {
namespace {


// returns the cached vector if it is configured like `like`, otherwise
// replaces it by a new one
template <typename VectorType>
VectorType* get_like(std::unique_ptr<LinOp>& cache, const VectorType* like)
{
    auto vec = dynamic_cast<VectorType*>(cache.get());
    if (!vec || vec->get_executor() != like->get_executor() ||
        vec->get_size() != like->get_size() ||
        gko::detail::get_local(vec)->get_size() !=
            gko::detail::get_local(like)->get_size()) {
        auto new_vec = gko::detail::create_with_config_of(like);
        vec = new_vec.get();
        cache = std::move(new_vec);
    }
    return vec;
}


}  // anonymous namespace


template <typename ValueType>
typename NullspaceProjection<ValueType>::parameters_type
NullspaceProjection<ValueType>::parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    auto params = NullspaceProjection::build();
    config::config_check_decorator config_check(config);
    if (auto& obj = config_check.get("preconditioner")) {
        params.with_preconditioner(
            config::parse_or_get_factory<const LinOpFactory>(obj, context,
                                                             td_for_child));
    }
    if (auto& obj = config_check.get("generated_preconditioner")) {
        params.with_generated_preconditioner(
            config::get_stored_obj<const LinOp>(obj, context));
    }
    if (auto& obj = config_check.get("nullspace")) {
        params.with_nullspace(
            config::get_stored_obj<const LinOp>(obj, context));
    }
    if (auto& obj = config_check.get("left_nullspace")) {
        params.with_left_nullspace(
            config::get_stored_obj<const LinOp>(obj, context));
    }
    return params;
}


template <typename ValueType>
NullspaceProjection<ValueType>::NullspaceProjection(
    std::shared_ptr<const Executor> exec)
    : LinOp(std::move(exec))
{}


template <typename ValueType>
NullspaceProjection<ValueType>::NullspaceProjection(
    const Factory* factory, std::shared_ptr<const LinOp> system_matrix)
    : LinOp(factory->get_executor(), gko::transpose(system_matrix->get_size())),
      parameters_{factory->get_parameters()}
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);
    auto exec = this->get_executor();
    const auto size = system_matrix->get_size()[0];
    if (parameters_.generated_preconditioner) {
        preconditioner_ = parameters_.generated_preconditioner;
    } else if (parameters_.preconditioner) {
        preconditioner_ = parameters_.preconditioner->generate(system_matrix);
    } else {
        preconditioner_ = matrix::Identity<ValueType>::create(exec, size);
    }
    GKO_ASSERT_EQUAL_DIMENSIONS(preconditioner_, this);
    if (preconditioner_->get_executor() != exec) {
        preconditioner_ = gko::clone(exec, preconditioner_);
    }
    nullspace_ = solver::detail::prepare_nullspace<ValueType>(
        parameters_.nullspace, "nullspace", system_matrix.get(), size, exec);
    left_nullspace_ = solver::detail::prepare_nullspace<ValueType>(
        parameters_.left_nullspace, "left_nullspace", system_matrix.get(), size,
        exec);
}


template <typename ValueType>
NullspaceProjection<ValueType>::NullspaceProjection(
    std::shared_ptr<const Executor> exec, dim<2> size,
    std::shared_ptr<const LinOp> preconditioner,
    std::shared_ptr<const solver::Nullspace<ValueType>> nullspace,
    std::shared_ptr<const solver::Nullspace<ValueType>> left_nullspace)
    : LinOp(std::move(exec), size),
      preconditioner_{std::move(preconditioner)},
      nullspace_{std::move(nullspace)},
      left_nullspace_{std::move(left_nullspace)}
{
    parameters_.with_generated_preconditioner(preconditioner_)
        .with_nullspace(nullspace_)
        .with_left_nullspace(left_nullspace_);
}


template <typename ValueType>
std::shared_ptr<const LinOp>
NullspaceProjection<ValueType>::get_preconditioner() const
{
    return preconditioner_;
}


template <typename ValueType>
std::shared_ptr<const solver::Nullspace<ValueType>>
NullspaceProjection<ValueType>::get_nullspace() const
{
    return nullspace_;
}


template <typename ValueType>
std::shared_ptr<const solver::Nullspace<ValueType>>
NullspaceProjection<ValueType>::get_left_nullspace() const
{
    return left_nullspace_;
}


template <typename ValueType>
std::unique_ptr<LinOp> NullspaceProjection<ValueType>::transpose() const
{
    // the transposed projectors would need the complex conjugate bases
    if (is_complex<ValueType>() && (nullspace_ || left_nullspace_)) {
        GKO_NOT_SUPPORTED(this);
    }
    // (P M Q)^T = Q^T M^T P^T, with symmetric projectors for real values
    return std::unique_ptr<NullspaceProjection>(new NullspaceProjection(
        this->get_executor(), gko::transpose(this->get_size()),
        share(as<Transposable>(preconditioner_.get())->transpose()),
        left_nullspace_, nullspace_));
}


template <typename ValueType>
std::unique_ptr<LinOp> NullspaceProjection<ValueType>::conj_transpose() const
{
    // (P M Q)^H = Q M^H P, since the projectors are Hermitian
    return std::unique_ptr<NullspaceProjection>(new NullspaceProjection(
        this->get_executor(), gko::transpose(this->get_size()),
        share(as<Transposable>(preconditioner_.get())->conj_transpose()),
        left_nullspace_, nullspace_));
}


template <typename ValueType>
void NullspaceProjection<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    experimental::precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_b, auto dense_x) {
            if (left_nullspace_) {
                auto projected = get_like(projected_input_.vec, dense_b);
                left_nullspace_->apply(dense_b, projected);
                preconditioner_->apply(projected, dense_x);
            } else {
                preconditioner_->apply(dense_b, dense_x);
            }
            if (nullspace_) {
                nullspace_->project(dense_x);
            }
        },
        b, x);
}


template <typename ValueType>
void NullspaceProjection<ValueType>::apply_impl(const LinOp* alpha,
                                                const LinOp* b,
                                                const LinOp* beta,
                                                LinOp* x) const
{
    experimental::precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto result = get_like(result_.vec, dense_x);
            this->apply_impl(dense_b, result);
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, result);
        },
        alpha, b, beta, x);
}


template <typename ValueType>
NullspaceProjection<ValueType>& NullspaceProjection<ValueType>::operator=(
    const NullspaceProjection& other)
{
    if (&other != this) {
        LinOp::operator=(other);
        auto exec = this->get_executor();
        parameters_ = other.parameters_;
        preconditioner_ = other.preconditioner_;
        nullspace_ = other.nullspace_;
        left_nullspace_ = other.left_nullspace_;
        if (other.get_executor() != exec) {
            if (preconditioner_) {
                preconditioner_ = gko::clone(exec, preconditioner_);
            }
            if (nullspace_) {
                nullspace_ = gko::clone(exec, nullspace_);
            }
            if (left_nullspace_) {
                left_nullspace_ = gko::clone(exec, left_nullspace_);
            }
        }
    }
    return *this;
}


template <typename ValueType>
NullspaceProjection<ValueType>& NullspaceProjection<ValueType>::operator=(
    NullspaceProjection&& other)
{
    if (&other != this) {
        *this = static_cast<const NullspaceProjection&>(other);
        other.preconditioner_ = nullptr;
        other.nullspace_ = nullptr;
        other.left_nullspace_ = nullptr;
    }
    return *this;
}


template <typename ValueType>
NullspaceProjection<ValueType>::NullspaceProjection(
    const NullspaceProjection& other)
    : NullspaceProjection(other.get_executor())
{
    *this = other;
}


template <typename ValueType>
NullspaceProjection<ValueType>::NullspaceProjection(NullspaceProjection&& other)
    : NullspaceProjection(other.get_executor())
{
    *this = std::move(other);
}


#define GKO_DECLARE_NULLSPACE_PROJECTION(ValueType) \
    class NullspaceProjection<ValueType>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_NULLSPACE_PROJECTION);


}  // namespace preconditioner
}  // namespace gko
