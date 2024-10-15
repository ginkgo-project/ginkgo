// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/stop/residual_norm.hpp"

#include <ginkgo/core/base/precision_dispatch.hpp>

#include "core/base/dispatch_helper.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/distributed/helpers.hpp"
#include "core/stop/residual_norm_kernels.hpp"


namespace gko {
namespace stop {
namespace residual_norm {
namespace {


GKO_REGISTER_OPERATION(residual_norm, residual_norm::residual_norm);


}  // anonymous namespace
}  // namespace residual_norm


namespace implicit_residual_norm {
namespace {


GKO_REGISTER_OPERATION(implicit_residual_norm,
                       implicit_residual_norm::implicit_residual_norm);


}  // anonymous namespace
}  // namespace implicit_residual_norm


template <typename ValueType>
bool any_is_complex()
{
    return false;
}


template <typename ValueType, typename LinOp, typename... Rest>
bool any_is_complex(const LinOp* in, Rest&&... rest)
{
#if GINKGO_BUILD_MPI
    bool is_complex_distributed = dynamic_cast<const ConvertibleTo<
        experimental::distributed::Vector<std::complex<double>>>*>(in);
#else
    bool is_complex_distributed = false;
#endif

    return is_complex<ValueType>() || is_complex_distributed ||
           dynamic_cast<
               const ConvertibleTo<matrix::Dense<std::complex<double>>>*>(in) ||
           any_is_complex<ValueType>(std::forward<Rest>(rest)...);
}


template <typename ValueType, typename Function, typename... LinOps>
void norm_dispatch(Function&& fn, LinOps*... linops)
{
#if GINKGO_BUILD_MPI
    if (gko::detail::is_distributed(linops...)) {
        if (any_is_complex<ValueType>(linops...)) {
            experimental::distributed::precision_dispatch<
                to_complex<ValueType>>(std::forward<Function>(fn), linops...);
        } else {
            experimental::distributed::precision_dispatch<ValueType>(
                std::forward<Function>(fn), linops...);
        }
    } else
#endif
    {
        if (any_is_complex<ValueType>(linops...)) {
            precision_dispatch<to_complex<ValueType>>(
                std::forward<Function>(fn), linops...);
        } else {
            precision_dispatch<ValueType>(std::forward<Function>(fn),
                                          linops...);
        }
    }
}

template <typename ValueType>
void ResidualNormBase<ValueType>::regenerate(const CriterionArgs& args)
{
    system_matrix_ = args.system_matrix;
    b_ = args.b;
    auto exec = this->get_executor();
    dim<2> column_size{1, b_->get_size()[1]};
    if (!this->starting_tau_ || starting_tau_->get_size() != column_size) {
        this->starting_tau_ = NormVector::create(exec, column_size);
        this->u_dense_tau_ =
            NormVector::create_with_config_of(this->starting_tau_);
    }
    switch (baseline_) {
    case mode::initial_resnorm: {
        if (args.initial_residual == nullptr) {
            if (args.system_matrix == nullptr || args.b == nullptr ||
                args.x == nullptr) {
                GKO_NOT_SUPPORTED(nullptr);
            } else {
                norm_dispatch<ValueType>(
                    [&](auto dense_b) {
                        this->set_cache_to(dense_b);
                        args.system_matrix->apply(neg_one_, args.x, one_,
                                                  cache_.intermediate);
                    },
                    args.b.get());
                norm_dispatch<ValueType>(
                    [&](auto dense_r) {
                        dense_r->compute_norm2(this->starting_tau_,
                                               reduction_tmp_);
                    },
                    cache_.intermediate.get());
            }
        } else {
            norm_dispatch<ValueType>(
                [&](auto dense_r) {
                    dense_r->compute_norm2(this->starting_tau_, reduction_tmp_);
                },
                args.initial_residual);
        }
        break;
    }
    case mode::rhs_norm: {
        if (args.b == nullptr) {
            GKO_NOT_SUPPORTED(nullptr);
        }
        norm_dispatch<ValueType>(
            [&](auto dense_r) {
                dense_r->compute_norm2(this->starting_tau_, reduction_tmp_);
            },
            args.b.get());
        break;
    }
    case mode::absolute: {
        if (args.b == nullptr) {
            GKO_NOT_SUPPORTED(nullptr);
        }
        this->starting_tau_->fill(gko::one<remove_complex<ValueType>>());
        break;
    }
    default:
        GKO_NOT_SUPPORTED(nullptr);
    }
}

template <typename ValueType>
ResidualNormBase<ValueType>::ResidualNormBase(
    std::shared_ptr<const gko::Executor> exec, const CriterionArgs& args,
    remove_complex<ValueType> reduction_factor, mode baseline)
    : EnablePolymorphicObject<ResidualNormBase, Criterion>(exec),
      reduction_factor_{reduction_factor},
      device_storage_{exec, 2},
      baseline_{baseline},
      system_matrix_{args.system_matrix},
      b_{args.b},
      one_{gko::initialize<Vector>({1}, exec)},
      neg_one_{gko::initialize<Vector>({-1}, exec)},
      reduction_tmp_{exec}
{
    this->regenerate(args);
}


template <typename ValueType>
bool ResidualNormBase<ValueType>::check_impl(
    uint8 stopping_id, bool set_finalized, array<stopping_status>* stop_status,
    bool* one_changed, const Criterion::Updater& updater)
{
    const NormVector* dense_tau;
    if (updater.residual_norm_ != nullptr) {
        dense_tau = as<NormVector>(updater.residual_norm_);
    } else if (updater.ignore_residual_check_) {
        // If solver already provide the residual norm, we will still store it.
        // Otherwise, we skip the residual check.
        return false;
    } else if (updater.residual_ != nullptr) {
        norm_dispatch<ValueType>(
            [&](auto dense_r) {
                dense_r->compute_norm2(u_dense_tau_, reduction_tmp_);
            },
            updater.residual_);
        dense_tau = u_dense_tau_.get();
    } else if (updater.solution_ != nullptr && system_matrix_ != nullptr &&
               b_ != nullptr) {
        auto exec = this->get_executor();
        norm_dispatch<ValueType>(
            [&](auto dense_b, auto dense_x) {
                this->set_cache_to(dense_b);
                system_matrix_->apply(neg_one_, dense_x, one_,
                                      cache_.intermediate);
                dynamic_cast<decltype(dense_b)>(cache_.intermediate.get())
                    ->compute_norm2(u_dense_tau_, reduction_tmp_);
            },
            b_.get(), updater.solution_);
        dense_tau = u_dense_tau_.get();
    } else {
        GKO_NOT_SUPPORTED(nullptr);
    }
    bool all_converged = true;

    this->get_executor()->run(residual_norm::make_residual_norm(
        dense_tau, starting_tau_.get(), reduction_factor_, stopping_id,
        set_finalized, stop_status, &device_storage_, &all_converged,
        one_changed));

    return all_converged;
}


template <typename ValueType>
template <typename VectorType>
void ResidualNormBase<ValueType>::set_cache_to(const VectorType* vec) const
{
    if (dynamic_cast<VectorType*>(cache_.intermediate.get()) == nullptr) {
        cache_.intermediate = VectorType::create_with_config_of(vec);
    }
    cache_.intermediate->copy_from(vec);
}

template <typename ValueType>
bool ImplicitResidualNorm<ValueType>::check_impl(
    uint8 stopping_id, bool set_finalized, array<stopping_status>* stop_status,
    bool* one_changed, const Criterion::Updater& updater)
{
    const Vector* dense_tau;
    if (updater.implicit_sq_residual_norm_ != nullptr) {
        dense_tau = as<Vector>(updater.implicit_sq_residual_norm_);
    } else {
        GKO_NOT_SUPPORTED(nullptr);
    }
    bool all_converged = true;

    this->get_executor()->run(
        implicit_residual_norm::make_implicit_residual_norm(
            dense_tau, this->starting_tau_.get(), this->reduction_factor_,
            stopping_id, set_finalized, stop_status, &this->device_storage_,
            &all_converged, one_changed));

    return all_converged;
}


#define GKO_DECLARE_RESIDUAL_NORM(_type) class ResidualNormBase<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_RESIDUAL_NORM);


#define GKO_DECLARE_IMPLICIT_RESIDUAL_NORM(_type) \
    class ImplicitResidualNorm<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IMPLICIT_RESIDUAL_NORM);


}  // namespace stop
}  // namespace gko
