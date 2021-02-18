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

#ifndef GKO_PUBLIC_CORE_STOP_RESIDUAL_NORM_HPP_
#define GKO_PUBLIC_CORE_STOP_RESIDUAL_NORM_HPP_


#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace stop {


/**
 * The mode for the residual norm criterion.
 *
 * - absolute:        Check for tolerance against residual norm.
 *                    $ || r || < \tau $
 *
 * - initial_resnorm: Check for tolerance relative to the initial residual norm.
 *                    $ \frac{|| r ||}{|| r_0||} < \tau $
 *
 * - rhs_resnorm:     Check for tolerance relative to the rhs norm.
 *                    $ \frac{|| r ||}{|| b ||} < \tau $
 *
 * @ingroup stop
 */
enum class mode { absolute, initial_resnorm, rhs_norm };


/**
 * The ResidualNormBase class provides a framework for stopping criteria
 * related to the residual norm. These criteria differ in the way they
 * initialize starting_tau_, so in the value they compare the
 * residual norm against.
 * The provided check_impl uses the actual residual to check for convergence.
 *
 * @ingroup stop
 */
template <typename ValueType>
class ResidualNormBase
    : public EnablePolymorphicObject<ResidualNormBase<ValueType>, Criterion> {
    friend class EnablePolymorphicObject<ResidualNormBase<ValueType>,
                                         Criterion>;

protected:
    using ComplexVector = matrix::Dense<to_complex<ValueType>>;
    using NormVector = matrix::Dense<remove_complex<ValueType>>;
    using Vector = matrix::Dense<ValueType>;
    bool check_impl(uint8 stoppingId, bool setFinalized,
                    Array<stopping_status> *stop_status, bool *one_changed,
                    const Criterion::Updater &updater) override;

    explicit ResidualNormBase(std::shared_ptr<const gko::Executor> exec)
        : EnablePolymorphicObject<ResidualNormBase, Criterion>(exec),
          device_storage_{exec, 2}
    {}

    explicit ResidualNormBase(std::shared_ptr<const gko::Executor> exec,
                              const CriterionArgs &args,
                              remove_complex<ValueType> reduction_factor,
                              mode baseline)
        : EnablePolymorphicObject<ResidualNormBase, Criterion>(exec),
          device_storage_{exec, 2},
          reduction_factor_{reduction_factor},
          baseline_{baseline}
    {
        switch (baseline_) {
        case mode::initial_resnorm: {
            if (args.initial_residual == nullptr) {
                GKO_NOT_SUPPORTED(nullptr);
            }
            this->starting_tau_ = NormVector::create(
                exec, dim<2>{1, args.initial_residual->get_size()[1]});
            if (dynamic_cast<const ComplexVector *>(args.initial_residual)) {
                auto dense_r = as<ComplexVector>(args.initial_residual);
                dense_r->compute_norm2(this->starting_tau_.get());
            } else {
                auto dense_r = as<Vector>(args.initial_residual);
                dense_r->compute_norm2(this->starting_tau_.get());
            }
            break;
        }
        case mode::rhs_norm: {
            if (args.b == nullptr) {
                GKO_NOT_SUPPORTED(nullptr);
            }
            this->starting_tau_ =
                NormVector::create(exec, dim<2>{1, args.b->get_size()[1]});
            if (dynamic_cast<const ComplexVector *>(args.b.get())) {
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

    remove_complex<ValueType> reduction_factor_{};
    std::unique_ptr<NormVector> starting_tau_{};
    std::unique_ptr<NormVector> u_dense_tau_{};
    /* Contains device side: all_converged and one_changed booleans */
    Array<bool> device_storage_;

private:
    mode baseline_{mode::rhs_norm};
};


/**
 * The ResidualNorm class is a stopping criterion which
 * stops the iteration process when the actual residual norm is below a
 * certain threshold relative to
 * 1. the norm of the right-hand side, norm(residual) / norm(right_hand_side)
 *                                                                  < threshold
 * 2. the initial residual, norm(residual) / norm(initial_residual) < threshold.
 * 3. one,  norm(residual) < threshold.
 *
 * For better performance, the checks are run on the executor
 * where the algorithm is executed.
 *
 * @note To use this stopping criterion there are some dependencies. The
 * constructor depends on either `b` or the `initial_residual` in order to
 * compute their norms. If this is not correctly provided, an exception
 * ::gko::NotSupported() is thrown.
 *
 * @ingroup stop
 */
template <typename ValueType = default_precision>
class ResidualNorm : public ResidualNormBase<ValueType> {
public:
    using ComplexVector = matrix::Dense<to_complex<ValueType>>;
    using NormVector = matrix::Dense<remove_complex<ValueType>>;
    using Vector = matrix::Dense<ValueType>;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Residual norm reduction factor
         */
        remove_complex<ValueType> GKO_FACTORY_PARAMETER_SCALAR(
            reduction_factor, static_cast<remove_complex<ValueType>>(1e-15));

        /**
         * The quantity the reduction is relative to. Choices include
         * "mode::rhs_norm", "mode::initial_resnorm" and "mode::absolute"
         */
        mode GKO_FACTORY_PARAMETER_SCALAR(baseline, mode::rhs_norm);
    };
    GKO_ENABLE_CRITERION_FACTORY(ResidualNorm<ValueType>, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    explicit ResidualNorm(std::shared_ptr<const gko::Executor> exec)
        : ResidualNormBase<ValueType>(exec)
    {}

    explicit ResidualNorm(const Factory *factory, const CriterionArgs &args)
        : ResidualNormBase<ValueType>(
              factory->get_executor(), args,
              factory->get_parameters().reduction_factor,
              factory->get_parameters().baseline),
          parameters_{factory->get_parameters()}
    {}
};


/**
 * The ImplicitResidualNorm class is a stopping criterion which
 * stops the iteration process when the implicit residual norm is below a
 * certain threshold relative to
 * 1. the norm of the right-hand side, implicit_resnorm / norm(right_hand_side)
 *                                                          < threshold
 * 2. the initial residual, implicit_resnorm / norm(initial_residual) <
 *                                                          < threshold.
 * 3. one, implicit_resnorm < threshold.
 *
 * @note To use this stopping criterion there are some dependencies. The
 * constructor depends on either `b` or the `initial_residual` in order to
 * compute their norms. If this is not correctly provided, an exception
 * ::gko::NotSupported() is thrown.
 *
 * @ingroup stop
 */
template <typename ValueType = default_precision>
class ImplicitResidualNorm : public ResidualNormBase<ValueType> {
public:
    using ComplexVector = matrix::Dense<to_complex<ValueType>>;
    using NormVector = matrix::Dense<remove_complex<ValueType>>;
    using Vector = matrix::Dense<ValueType>;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Implicit Residual norm goal
         */
        remove_complex<ValueType> GKO_FACTORY_PARAMETER_SCALAR(
            reduction_factor, static_cast<remove_complex<ValueType>>(1e-15));

        /**
         * The quantity the reduction is relative to. Choices include
         * "mode::rhs_norm", "mode::initial_resnorm" and "mode::absolute"
         */
        mode GKO_FACTORY_PARAMETER_SCALAR(baseline, mode::rhs_norm);
    };
    GKO_ENABLE_CRITERION_FACTORY(ImplicitResidualNorm<ValueType>, parameters,
                                 Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    // check_impl needs to be overwritten again since we focus on the implicit
    // residual here
    bool check_impl(uint8 stoppingId, bool setFinalized,
                    Array<stopping_status> *stop_status, bool *one_changed,
                    const Criterion::Updater &updater) override;

    explicit ImplicitResidualNorm(std::shared_ptr<const gko::Executor> exec)
        : ResidualNormBase<ValueType>(exec)
    {}

    explicit ImplicitResidualNorm(const Factory *factory,
                                  const CriterionArgs &args)
        : ResidualNormBase<ValueType>(
              factory->get_executor(), args,
              factory->get_parameters().reduction_factor,
              factory->get_parameters().baseline),
          parameters_{factory->get_parameters()}
    {}
};


/**
 * The ResidualNormReduction class is a stopping criterion which stops the
 * iteration process when the residual norm is below a certain
 * threshold relative to the norm of the initial residual, i.e. when
 * norm(residual) / norm(initial_residual) < threshold.
 * For better performance, the checks are run thanks to kernels on
 * the executor where the algorithm is executed.
 *
 * @note To use this stopping criterion there are some dependencies. The
 * constructor depends on `initial_residual` in order to compute the first
 * relative residual norm. The check method depends on either the
 * `residual_norm` or the `residual` being set. When any of those is not
 * correctly provided, an exception ::gko::NotSupported() is thrown.
 *
 * @deprecated Please use the class ResidualNorm with the factory parameter
 *             baseline = mode::initial_resnorm
 *
 * @ingroup stop
 */
template <typename ValueType = default_precision>
class ResidualNormReduction : public ResidualNormBase<ValueType> {
public:
    using ComplexVector = matrix::Dense<to_complex<ValueType>>;
    using NormVector = matrix::Dense<remove_complex<ValueType>>;
    using Vector = matrix::Dense<ValueType>;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Factor by which the residual norm will be reduced
         */
        remove_complex<ValueType> GKO_FACTORY_PARAMETER_SCALAR(
            reduction_factor, static_cast<remove_complex<ValueType>>(1e-15));
    };
    GKO_ENABLE_CRITERION_FACTORY(ResidualNormReduction<ValueType>, parameters,
                                 Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    explicit ResidualNormReduction(std::shared_ptr<const gko::Executor> exec)
        : ResidualNormBase<ValueType>(exec)
    {}

    explicit ResidualNormReduction(const Factory *factory,
                                   const CriterionArgs &args)
        : ResidualNormBase<ValueType>(
              factory->get_executor(), args,
              factory->get_parameters().reduction_factor,
              mode::initial_resnorm),
          parameters_{factory->get_parameters()}
    {}
};


/**
 * The RelativeResidualNorm class is a stopping criterion which stops the
 * iteration process when the residual norm is below a certain
 * threshold relative to the norm of the right-hand side, i.e. when
 * norm(residual) / norm(right_hand_side) < threshold.
 * For better performance, the checks are run thanks to kernels on
 * the executor where the algorithm is executed.
 *
 * @note To use this stopping criterion there are some dependencies. The
 * constructor depends on `b` in order to compute the norm of the
 * right-hand side. If this is not correctly provided, an exception
 * ::gko::NotSupported() is thrown.
 *
 * @deprecated Please use the class ResidualNorm with the factory parameter
 *             baseline = mode::rhs_norm
 *
 * @ingroup stop
 */
template <typename ValueType = default_precision>
class RelativeResidualNorm : public ResidualNormBase<ValueType> {
public:
    using ComplexVector = matrix::Dense<to_complex<ValueType>>;
    using NormVector = matrix::Dense<remove_complex<ValueType>>;
    using Vector = matrix::Dense<ValueType>;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Relative residual norm goal
         */
        remove_complex<ValueType> GKO_FACTORY_PARAMETER_SCALAR(
            tolerance, static_cast<remove_complex<ValueType>>(1e-15));
    };
    GKO_ENABLE_CRITERION_FACTORY(RelativeResidualNorm<ValueType>, parameters,
                                 Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    explicit RelativeResidualNorm(std::shared_ptr<const gko::Executor> exec)
        : ResidualNormBase<ValueType>(exec)
    {}

    explicit RelativeResidualNorm(const Factory *factory,
                                  const CriterionArgs &args)
        : ResidualNormBase<ValueType>(factory->get_executor(), args,
                                      factory->get_parameters().tolerance,
                                      mode::rhs_norm),
          parameters_{factory->get_parameters()}
    {}
};


/**
 * The AbsoluteResidualNorm class is a stopping criterion which stops the
 * iteration process when the residual norm is below a certain
 * threshold, i.e. when norm(residual) / threshold.
 * For better performance, the checks are run thanks to kernels on
 * the executor where the algorithm is executed.
 *
 * @note To use this stopping criterion there are some dependencies. The
 * constructor depends on `b` in order to get the number of right-hand sides.
 * If this is not correctly provided, an exception ::gko::NotSupported()
 * is thrown.
 *
 * @deprecated Please use the class ResidualNorm with the factory parameter
 *             baseline = mode::absolute
 *
 * @ingroup stop
 */
template <typename ValueType = default_precision>
class AbsoluteResidualNorm : public ResidualNormBase<ValueType> {
public:
    using NormVector = matrix::Dense<remove_complex<ValueType>>;
    using Vector = matrix::Dense<ValueType>;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Absolute residual norm goal
         */
        remove_complex<ValueType> GKO_FACTORY_PARAMETER_SCALAR(
            tolerance, static_cast<remove_complex<ValueType>>(1e-15));
    };
    GKO_ENABLE_CRITERION_FACTORY(AbsoluteResidualNorm<ValueType>, parameters,
                                 Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    explicit AbsoluteResidualNorm(std::shared_ptr<const gko::Executor> exec)
        : ResidualNormBase<ValueType>(exec)
    {}

    explicit AbsoluteResidualNorm(const Factory *factory,
                                  const CriterionArgs &args)
        : ResidualNormBase<ValueType>(factory->get_executor(), args,
                                      factory->get_parameters().tolerance,
                                      mode::absolute),
          parameters_{factory->get_parameters()}
    {}
};


}  // namespace stop
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_STOP_RESIDUAL_NORM_HPP_
