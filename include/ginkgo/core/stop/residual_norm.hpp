// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_STOP_RESIDUAL_NORM_HPP_
#define GKO_PUBLIC_CORE_STOP_RESIDUAL_NORM_HPP_


#include <limits>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/multivector.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace stop {


/**
 * The mode for the residual norm criterion.
 *
 * - absolute:        Check for tolerance against residual norm.
 *                    \f$ || r || \leq \tau \f$
 *
 * - initial_resnorm: Check for tolerance relative to the initial residual norm.
 *                    \f$ || r || \leq \tau \times || r_0 || \f$
 *
 * - rhs_norm:        Check for tolerance relative to the rhs norm.
 *                    \f$ || r || \leq \tau \times || b || \f$
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
class ResidualNormBase : public Criterion {
    GKO_ASSERT_SUPPORTED_VALUE_TYPE;

protected:
    using absolute_type = remove_complex<ValueType>;
    using ComplexVector = matrix::MultiVector<to_complex<ValueType>>;
    using NormVector = matrix::MultiVector<absolute_type>;
    using Vector = matrix::MultiVector<ValueType>;
    bool check_impl(uint8 stoppingId, bool setFinalized,
                    array<stopping_status>* stop_status, bool* one_changed,
                    const Criterion::Updater& updater) override;

    explicit ResidualNormBase(std::shared_ptr<const gko::Executor> exec)
        : Criterion(exec), device_storage_{exec, 2}
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
    // workspace for reduction
    mutable gko::array<char> reduction_tmp_;
};


/**
 * Stopping criterion based on the explicit residual norm
 * \f$ \| r_k \| = \| b - A x_k \| \f$. The iteration halts once
 * \f$ \| r_k \| \le \tau \cdot \beta \f$, where \f$\tau\f$ is the
 * factory parameter `reduction_factor` and the baseline \f$\beta\f$ is
 * selected by the `baseline` factory parameter:
 *
 * - `mode::rhs_norm` (default) — relative to the right-hand side:
 *   \f$ \beta = \| b \| \f$, so the condition is
 *   \f$ \| r_k \| \le \tau \| b \| \f$.
 * - `mode::initial_resnorm` — relative to the initial residual:
 *   \f$ \beta = \| r_0 \| \f$, so the condition is
 *   \f$ \| r_k \| \le \tau \| r_0 \| \f$.
 * - `mode::absolute` — absolute threshold:
 *   \f$ \beta = 1 \f$, so the condition is \f$ \| r_k \| \le \tau \f$.
 *
 * Per-iteration, the criterion prefers a pre-computed residual norm
 * passed by the solver (via `Updater::residual_norm`); otherwise it
 * falls back to computing the 2-norm of the residual vector itself
 * (via `Updater::residual`), which costs one global reduction. When
 * even cheaper checks are needed and the solver maintains an internal
 * squared-norm estimate, use \ref ImplicitResidualNorm instead.
 *
 * @note Baseline prerequisites at construction time:
 *       - `mode::rhs_norm` requires the right-hand side \f$ b \f$.
 *       - `mode::initial_resnorm` requires either the initial residual
 *         \f$ r_0 \f$ explicitly, or the triple
 *         \f$ (A, b, x_0) \f$ from which it is computed as
 *         \f$ r_0 = b - A x_0 \f$.
 *       - `mode::absolute` requires \f$ b \f$ as well — to determine the
 *         number of right-hand sides; its baseline is then filled with
 *         ones.
 *       If the required arguments are missing, ::gko::NotSupported() is
 *       thrown.
 *
 * @ingroup stop
 */
template <typename ValueType = default_precision>
class ResidualNorm : public ResidualNormBase<ValueType> {
public:
    using ComplexVector = matrix::MultiVector<to_complex<ValueType>>;
    using NormVector = matrix::MultiVector<remove_complex<ValueType>>;
    using Vector = matrix::MultiVector<ValueType>;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Residual norm reduction factor
         */
        remove_complex<ValueType> reduction_factor{
            5 * std ::numeric_limits<remove_complex<ValueType>>::epsilon()};

        parameters_type& with_reduction_factor(remove_complex<ValueType> value)
        {
            this->reduction_factor = value;
            return *this;
        }

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
 * Stopping criterion based on a solver-maintained squared residual-norm
 * estimate \f$ \rho_k^2 \approx \| r_k \|^2 \f$. Several Krylov methods
 * (CG, BiCGSTAB, …) already compute this quantity per iteration as a
 * by-product of their inner-product recurrences, so the criterion can
 * check convergence without computing a norm of its own — saving the
 * global reduction the explicit \ref ResidualNorm form needs when the
 * solver only passes \f$ r_k \f$. The iteration halts once
 * \f$ \rho_k \le \tau \cdot \beta \f$, with the same `reduction_factor`
 * and `baseline` factory parameters as \ref ResidualNorm:
 *
 * - `mode::rhs_norm` (default) — \f$ \rho_k \le \tau \| b \| \f$.
 * - `mode::initial_resnorm` — \f$ \rho_k \le \tau \| r_0 \| \f$.
 * - `mode::absolute` — \f$ \rho_k \le \tau \f$.
 *
 * Because \f$ \rho_k^2 \f$ is updated by short recurrences rather than
 * recomputed from \f$ b - A x_k \f$ each step, it can drift from the
 * true residual norm on long runs in finite precision; pair this
 * criterion with an \ref Iteration cap when that matters.
 *
 * @note The solver must pass the squared estimate through
 *       `Updater::implicit_sq_residual_norm` on every check — there is
 *       no fallback path. If it is missing, ::gko::NotSupported() is
 *       thrown at check time.
 *
 * @note Baseline prerequisites at construction time mirror
 *       \ref ResidualNorm:
 *       - `mode::rhs_norm` requires \f$ b \f$.
 *       - `mode::initial_resnorm` requires either \f$ r_0 \f$ explicitly
 *         or the triple \f$ (A, b, x_0) \f$ to derive it.
 *       - `mode::absolute` requires \f$ b \f$ for sizing the per-RHS
 *         baseline.
 *       If the required arguments are missing, ::gko::NotSupported() is
 *       thrown.
 *
 * @ingroup stop
 */
template <typename ValueType = default_precision>
class ImplicitResidualNorm : public ResidualNormBase<ValueType> {
public:
    using ComplexVector = matrix::MultiVector<to_complex<ValueType>>;
    using NormVector = matrix::MultiVector<remove_complex<ValueType>>;
    using Vector = matrix::MultiVector<ValueType>;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Implicit Residual norm goal
         */
        remove_complex<ValueType> reduction_factor{
            5 * std ::numeric_limits<remove_complex<ValueType>>::epsilon()};

        parameters_type& with_reduction_factor(remove_complex<ValueType> value)
        {
            this->reduction_factor = value;
            return *this;
        }

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


/**
 * Creates the precursor to a ResidualNorm stopping criterion factory, to be
 * used in conjunction with `.with_criteria(...)` function calls when building a
 * solver factory. This stopping criterion will stop the iteration after the
 * residual norm has decreased below the specified value or by the specified
 * amount.
 *
 * Full usage example: Stop after 100 iterations or when the absolute residual
 * norm is below \f$10^{-10}\f$, whichever happens first.
 * ```cpp
 * auto factory = gko::solver::Cg<double>::build()
 *                    .with_criteria(
 *                        gko::stop::max_iters(100),
 *                        gko::stop::absolute_residual_norm(1e-10))
 *                    .on(exec);
 * ```
 *
 * @param tolerance  the value the residual norm needs to be below.
 *     With residual \f$r\f$, initial guess \f$x_0\f$, right-hand side
 *     \f$b\f$ and matrix \f$A\f$, `absolute` means the exact value of the
 *     norm \f$||r||\f$, `relative` means the norm relative to the right-hand
 *     side \f$||r||/||b||\f$, `initial` means the norm relative to the
 *     initial residual \f$||r||/||b - A x_0||\f$.
 *     An implicit stopping criterion is only available with some solvers, and
 *     refers to either the energy norm \f$||r||_A\f$ in short-recurrence
 *     solvers like Cg or the euclidian norm \f$||r||\f$ in solvers like
 *     GMRES.
 *     Implicit residual norms are cheaper to compute, but may be less precise
 *     due to accumulating rounding errors.
 * @return a deferred_factory_parameter that can be passed to the
 *         `with_criteria` function when building a solver.
 */
deferred_factory_parameter<CriterionFactory> absolute_residual_norm(
    double tolerance);

/** @copydoc absolute_residual_norm */
deferred_factory_parameter<CriterionFactory> relative_residual_norm(
    double tolerance);

/** @copydoc absolute_residual_norm */
deferred_factory_parameter<CriterionFactory> initial_residual_norm(
    double tolerance);

/** @copydoc absolute_residual_norm */
deferred_factory_parameter<CriterionFactory> absolute_implicit_residual_norm(
    double tolerance);

/** @copydoc absolute_residual_norm */
deferred_factory_parameter<CriterionFactory> relative_implicit_residual_norm(
    double tolerance);

/** @copydoc absolute_residual_norm */
deferred_factory_parameter<CriterionFactory> initial_implicit_residual_norm(
    double tolerance);


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
    using ComplexVector = matrix::MultiVector<to_complex<ValueType>>;
    using NormVector = matrix::MultiVector<remove_complex<ValueType>>;
    using Vector = matrix::MultiVector<ValueType>;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Factor by which the residual norm will be reduced
         */
        remove_complex<ValueType> reduction_factor{
            5 * std ::numeric_limits<remove_complex<ValueType>>::epsilon()};

        parameters_type& with_reduction_factor(remove_complex<ValueType> value)
        {
            this->reduction_factor = value;
            return *this;
        }
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
    using ComplexVector = matrix::MultiVector<to_complex<ValueType>>;
    using NormVector = matrix::MultiVector<remove_complex<ValueType>>;
    using Vector = matrix::MultiVector<ValueType>;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Relative residual norm goal
         */
        remove_complex<ValueType> tolerance{
            5 * std ::numeric_limits<remove_complex<ValueType>>::epsilon()};


        parameters_type& with_tolerance(remove_complex<ValueType> value)
        {
            this->tolerance = value;
            return *this;
        }
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
    using NormVector = matrix::MultiVector<remove_complex<ValueType>>;
    using Vector = matrix::MultiVector<ValueType>;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Absolute residual norm goal
         */
        remove_complex<ValueType> tolerance{
            5 * std ::numeric_limits<remove_complex<ValueType>>::epsilon()};

        parameters_type& with_tolerance(remove_complex<ValueType> value)
        {
            this->tolerance = value;
            return *this;
        }
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
