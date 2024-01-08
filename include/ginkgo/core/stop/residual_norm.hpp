// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
 *                    $ || r || \leq \tau $
 *
 * - initial_resnorm: Check for tolerance relative to the initial residual norm.
 *                    $ || r || \leq \tau \times || r_0|| $
 *
 * - rhs_norm:        Check for tolerance relative to the rhs norm.
 *                    $ || r || \leq \tau \times || b || $
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
    using absolute_type = remove_complex<ValueType>;
    using ComplexVector = matrix::Dense<to_complex<ValueType>>;
    using NormVector = matrix::Dense<absolute_type>;
    using Vector = matrix::Dense<ValueType>;
    bool check_impl(uint8 stoppingId, bool setFinalized,
                    array<stopping_status>* stop_status, bool* one_changed,
                    const Criterion::Updater& updater) override;

    explicit ResidualNormBase(std::shared_ptr<const gko::Executor> exec)
        : EnablePolymorphicObject<ResidualNormBase, Criterion>(exec),
          device_storage_{exec, 2}
    {}

    explicit ResidualNormBase(std::shared_ptr<const gko::Executor> exec,
                              const CriterionArgs& args,
                              absolute_type reduction_factor, mode baseline);

    remove_complex<ValueType> reduction_factor_{};
    std::unique_ptr<NormVector> starting_tau_{};
    std::unique_ptr<NormVector> u_dense_tau_{};
    /* Contains device side: all_converged and one_changed booleans */
    array<bool> device_storage_;

private:
    mode baseline_{mode::rhs_norm};
    std::shared_ptr<const LinOp> system_matrix_{};
    std::shared_ptr<const LinOp> b_{};
    /* one/neg_one for residual computation */
    std::shared_ptr<const Vector> one_{};
    std::shared_ptr<const Vector> neg_one_{};
};


/**
 * The ResidualNorm class is a stopping criterion which
 * stops the iteration process when the actual residual norm is below a
 * certain threshold relative to
 * 1. the norm of the right-hand side, norm(residual) $\leq$ < threshold *
 *    norm(right_hand_side).
 * 2. the initial residual, norm(residual) $\leq$ threshold *
 *    norm(initial_residual).
 * 3. one,  norm(residual) $\leq$ threshold.
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

    explicit ResidualNorm(const Factory* factory, const CriterionArgs& args)
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
 * 1. the norm of the right-hand side, implicit_resnorm $\leq$ < threshold *
 * norm(right_hand_side)
 * 2. the initial residual, implicit_resnorm $\leq$ threshold *
 * norm(initial_residual) .
 * 3. one,  implicit_resnorm $\leq$ threshold.
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
                    array<stopping_status>* stop_status, bool* one_changed,
                    const Criterion::Updater& updater) override;

    explicit ImplicitResidualNorm(std::shared_ptr<const gko::Executor> exec)
        : ResidualNormBase<ValueType>(exec)
    {}

    explicit ImplicitResidualNorm(const Factory* factory,
                                  const CriterionArgs& args)
        : ResidualNormBase<ValueType>(
              factory->get_executor(), args,
              factory->get_parameters().reduction_factor,
              factory->get_parameters().baseline),
          parameters_{factory->get_parameters()}
    {}
};


// The following classes are deprecated, but they internally reference
// themselves. To reduce unnecessary warnings, we disable deprecation warnings
// for the definition of these classes.
GKO_BEGIN_DISABLE_DEPRECATION_WARNINGS


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
class GKO_DEPRECATED(
    "Please use the class ResidualNorm with the factory parameter baseline = "
    "mode::initial_resnorm") ResidualNormReduction
    : public ResidualNormBase<ValueType> {
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

    explicit ResidualNormReduction(const Factory* factory,
                                   const CriterionArgs& args)
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
class GKO_DEPRECATED(
    "Please use the class ResidualNorm with the factory parameter baseline = "
    "mode::rhs_norm") RelativeResidualNorm
    : public ResidualNormBase<ValueType> {
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

    explicit RelativeResidualNorm(const Factory* factory,
                                  const CriterionArgs& args)
        : ResidualNormBase<ValueType>(factory->get_executor(), args,
                                      factory->get_parameters().tolerance,
                                      mode::rhs_norm),
          parameters_{factory->get_parameters()}
    {}
};


/**
 * The AbsoluteResidualNorm class is a stopping criterion which stops the
 * iteration process when the residual norm is below a certain
 * threshold, i.e. when norm(residual) < threshold.
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
class GKO_DEPRECATED(
    "Please use the class ResidualNorm with the factory parameter baseline = "
    "mode::absolute") AbsoluteResidualNorm
    : public ResidualNormBase<ValueType> {
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

    explicit AbsoluteResidualNorm(const Factory* factory,
                                  const CriterionArgs& args)
        : ResidualNormBase<ValueType>(factory->get_executor(), args,
                                      factory->get_parameters().tolerance,
                                      mode::absolute),
          parameters_{factory->get_parameters()}
    {}
};


GKO_END_DISABLE_DEPRECATION_WARNINGS


}  // namespace stop
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_STOP_RESIDUAL_NORM_HPP_
