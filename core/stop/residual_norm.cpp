/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/components/fill_array_kernels.hpp"
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
ResidualNormBase<ValueType>::ResidualNormBase(
    std::shared_ptr<const gko::Executor> exec)
    : EnablePolymorphicObject<ResidualNormBase, Criterion>(exec),
      device_storage_{exec, 2}
{}


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
      neg_one_{gko::initialize<Vector>({-1}, exec)}
{
    switch (baseline_) {
    case mode::initial_resnorm: {
        if (args.initial_residual == nullptr) {
            if (args.system_matrix == nullptr || args.b == nullptr ||
                args.x == nullptr) {
                GKO_NOT_SUPPORTED(nullptr);
            } else {
                this->starting_tau_ =
                    NormVector::create(exec, dim<2>{1, args.b->get_size()[1]});
                auto b_clone = share(args.b->clone());
                args.system_matrix->apply(neg_one_.get(), args.x, one_.get(),
                                          b_clone.get());
                if (auto vec = std::dynamic_pointer_cast<const ComplexVector>(
                        b_clone)) {
                    vec->compute_norm2(this->starting_tau_.get());
                } else if (auto vec = std::dynamic_pointer_cast<const Vector>(
                               b_clone)) {
                    vec->compute_norm2(this->starting_tau_.get());
                } else {
                    GKO_NOT_SUPPORTED(nullptr);
                }
            }
        } else {
            this->starting_tau_ = NormVector::create(
                exec, dim<2>{1, args.initial_residual->get_size()[1]});
            if (dynamic_cast<const ComplexVector*>(args.initial_residual)) {
                auto dense_r = as<ComplexVector>(args.initial_residual);
                dense_r->compute_norm2(this->starting_tau_.get());
            } else {
                auto dense_r = as<Vector>(args.initial_residual);
                dense_r->compute_norm2(this->starting_tau_.get());
            }
        }
        break;
    }
    case mode::rhs_norm: {
        if (args.b == nullptr) {
            GKO_NOT_SUPPORTED(nullptr);
        }
        this->starting_tau_ =
            NormVector::create(exec, dim<2>{1, args.b->get_size()[1]});
        if (dynamic_cast<const ComplexVector*>(args.b.get())) {
            auto dense_rhs = as<ComplexVector>(args.b);
            dense_rhs->compute_norm2(this->starting_tau_.get());
        } else {
            auto dense_rhs = as<Vector>(args.b);
            dense_rhs->compute_norm2(this->starting_tau_.get());
        }
        break;
    }
    case mode::absolute: {
        if (args.b == nullptr) {
            GKO_NOT_SUPPORTED(nullptr);
        }
        this->starting_tau_ =
            NormVector::create(exec, dim<2>{1, args.b->get_size()[1]});
        this->starting_tau_->fill(gko::one<remove_complex<ValueType>>());
        break;
    }
    default:
        GKO_NOT_SUPPORTED(nullptr);
    }
    this->u_dense_tau_ =
        NormVector::create_with_config_of(this->starting_tau_.get());
}


template <typename ValueType>
bool ResidualNormBase<ValueType>::check_impl(
    uint8 stopping_id, bool set_finalized, array<stopping_status>* stop_status,
    bool* one_changed, const Criterion::Updater& updater)
{
    const NormVector* dense_tau;
    if (updater.residual_norm_ != nullptr) {
        dense_tau = as<NormVector>(updater.residual_norm_);
    } else if (updater.residual_ != nullptr) {
        if (dynamic_cast<const ComplexVector*>(updater.residual_)) {
            auto* dense_r = as<ComplexVector>(updater.residual_);
            dense_r->compute_norm2(u_dense_tau_.get());
        } else {
            auto* dense_r = as<Vector>(updater.residual_);
            dense_r->compute_norm2(u_dense_tau_.get());
        }
        dense_tau = u_dense_tau_.get();
    } else if (updater.solution_ != nullptr && system_matrix_ != nullptr &&
               b_ != nullptr) {
        auto exec = this->get_executor();
        // when LinOp is real but rhs is complex, we use real view on complex,
        // so it still uses the same type of scalar in apply.
        if (auto vec_b = std::dynamic_pointer_cast<const Vector>(b_)) {
            auto dense_r = vec_b->clone();
            system_matrix_->apply(neg_one_.get(), updater.solution_, one_.get(),
                                  dense_r.get());
            dense_r->compute_norm2(u_dense_tau_.get());
        } else if (auto vec_b =
                       std::dynamic_pointer_cast<const ComplexVector>(b_)) {
            auto dense_r = vec_b->clone();
            system_matrix_->apply(neg_one_.get(), updater.solution_, one_.get(),
                                  dense_r.get());
            dense_r->compute_norm2(u_dense_tau_.get());
        } else {
            GKO_NOT_SUPPORTED(nullptr);
        }
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


template <typename ValueType>
ResidualNorm<ValueType>::ResidualNorm(std::shared_ptr<const gko::Executor> exec)
    : ResidualNormBase<ValueType>(exec)
{}


template <typename ValueType>
ResidualNorm<ValueType>::ResidualNorm(const Factory* factory,
                                      const CriterionArgs& args)
    : ResidualNormBase<ValueType>(factory->get_executor(), args,
                                  factory->get_parameters().reduction_factor,
                                  factory->get_parameters().baseline),
      parameters_{factory->get_parameters()}
{}


template <typename ValueType>
ImplicitResidualNorm<ValueType>::ImplicitResidualNorm(
    std::shared_ptr<const gko::Executor> exec)
    : ResidualNormBase<ValueType>(exec)
{}


template <typename ValueType>
ImplicitResidualNorm<ValueType>::ImplicitResidualNorm(const Factory* factory,
                                                      const CriterionArgs& args)
    : ResidualNormBase<ValueType>(factory->get_executor(), args,
                                  factory->get_parameters().reduction_factor,
                                  factory->get_parameters().baseline),
      parameters_{factory->get_parameters()}
{}


// The following classes are deprecated, but they internally reference
// themselves. To reduce unnecessary warnings, we disable deprecation warnings
// for the definition of these classes.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_BUILD) || defined(__INTEL_LLVM_COMPILER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif


template <typename ValueType>
ResidualNormReduction<ValueType>::ResidualNormReduction(
    std::shared_ptr<const gko::Executor> exec)
    : ResidualNormBase<ValueType>(exec)
{}


template <typename ValueType>
ResidualNormReduction<ValueType>::ResidualNormReduction(
    const Factory* factory, const CriterionArgs& args)
    : ResidualNormBase<ValueType>(factory->get_executor(), args,
                                  factory->get_parameters().reduction_factor,
                                  mode::initial_resnorm),
      parameters_{factory->get_parameters()}
{}


template <typename ValueType>
RelativeResidualNorm<ValueType>::RelativeResidualNorm(
    std::shared_ptr<const gko::Executor> exec)
    : ResidualNormBase<ValueType>(exec)
{}


template <typename ValueType>
RelativeResidualNorm<ValueType>::RelativeResidualNorm(const Factory* factory,
                                                      const CriterionArgs& args)
    : ResidualNormBase<ValueType>(factory->get_executor(), args,
                                  factory->get_parameters().tolerance,
                                  mode::rhs_norm),
      parameters_{factory->get_parameters()}
{}


template <typename ValueType>
AbsoluteResidualNorm<ValueType>::AbsoluteResidualNorm(
    std::shared_ptr<const gko::Executor> exec)
    : ResidualNormBase<ValueType>(exec)
{}


template <typename ValueType>
AbsoluteResidualNorm<ValueType>::AbsoluteResidualNorm(const Factory* factory,
                                                      const CriterionArgs& args)
    : ResidualNormBase<ValueType>(factory->get_executor(), args,
                                  factory->get_parameters().tolerance,
                                  mode::absolute),
      parameters_{factory->get_parameters()}
{}


#define GKO_DECLARE_RESIDUAL_NORM_BASE(_type) class ResidualNormBase<_type>
#define GKO_DECLARE_RESIDUAL_NORM(_type) class ResidualNorm<_type>
#define GKO_DECLARE_RESIDUAL_NORM_REDUCTION(_type) \
    class ResidualNormReduction<_type>
#define GKO_DECLARE_REL_RESIDUAL_NORM(_type) class RelativeResidualNorm<_type>
#define GKO_DECLARE_ABS_RESIDUAL_NORM(_type) class AbsoluteResidualNorm<_type>
#define GKO_DECLARE_IMPLICIT_RESIDUAL_NORM(_type) \
    class ImplicitResidualNorm<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_RESIDUAL_NORM_BASE);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_RESIDUAL_NORM);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_RESIDUAL_NORM_REDUCTION);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_REL_RESIDUAL_NORM);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_ABS_RESIDUAL_NORM);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IMPLICIT_RESIDUAL_NORM);


#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#elif defined(_MSC_BUILD) || defined(__INTEL_LLVM_COMPILER)
#pragma warning(pop)
#endif


}  // namespace stop
}  // namespace gko
