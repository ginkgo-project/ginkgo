/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#ifndef GKO_CORE_STOP_RESIDUAL_NORM_HPP_
#define GKO_CORE_STOP_RESIDUAL_NORM_HPP_


#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace stop {


/**
 * The ResidualNorm class provides a framework for stopping criteria
 * related to the residual norm. These criteria differ in the way they
 * initialize starting_tau_, so in the value they compare the
 * residual norm against.
 *
 * @ingroup stop
 */
template <typename ValueType = default_precision>
class ResidualNorm
    : public EnablePolymorphicObject<ResidualNorm<ValueType>, Criterion> {
    friend class EnablePolymorphicObject<ResidualNorm<ValueType>, Criterion>;

public:
    using NormVector = matrix::Dense<remove_complex<ValueType>>;
    using Vector = matrix::Dense<ValueType>;

protected:
    bool check_impl(uint8 stoppingId, bool setFinalized,
                    Array<stopping_status> *stop_status, bool *one_changed,
                    const Criterion::Updater &) override;

    explicit ResidualNorm(std::shared_ptr<const gko::Executor> exec)
        : EnablePolymorphicObject<ResidualNorm, Criterion>(exec),
          device_storage_{exec, 2}
    {}

    explicit ResidualNorm(std::shared_ptr<const gko::Executor> exec,
                          remove_complex<ValueType> tolerance)
        : EnablePolymorphicObject<ResidualNorm, Criterion>(exec),
          device_storage_{exec, 2},
          tolerance_{tolerance}
    {}

    std::unique_ptr<NormVector> starting_tau_{};
    std::unique_ptr<NormVector> u_dense_tau_{};

private:
    remove_complex<ValueType> tolerance_{};
    /* Contains device side: all_converged and one_changed booleans */
    Array<bool> device_storage_;
};


/**
 * The ResidualNormReduction class is a stopping criterion which stops the
 * iteration process when the residual norm is below a certain
 * threshold relative to the norm of the initial residual.
 * For better performance, the checks are run thanks to kernels on
 * the executor where the algorithm is executed.
 *
 * @note To use this stopping criterion there are some dependencies. The
 * constructor depends on `initial_residual` in order to compute the first
 * relative residual norm. The check method depends on either the
 * `residual_norm` or the `residual` being set. When any of those is not
 * correctly provided, an exception ::gko::NotSupported() is thrown.
 *
 * @ingroup stop
 */
template <typename ValueType = default_precision>
class ResidualNormReduction : public ResidualNorm<ValueType> {
public:
    using NormVector = matrix::Dense<remove_complex<ValueType>>;
    using Vector = matrix::Dense<ValueType>;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Factor by which the residual norm will be reduced
         */
        remove_complex<ValueType> GKO_FACTORY_PARAMETER(reduction_factor,
                                                        1e-15);
    };
    GKO_ENABLE_CRITERION_FACTORY(ResidualNormReduction<ValueType>, parameters,
                                 Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    explicit ResidualNormReduction(std::shared_ptr<const gko::Executor> exec)
        : ResidualNorm<ValueType>(exec)
    {}

    explicit ResidualNormReduction(const Factory *factory,
                                   const CriterionArgs &args)
        : ResidualNorm<ValueType>(factory->get_executor(),
                                  factory->get_parameters().reduction_factor),
          parameters_{factory->get_parameters()}
    {
        if (args.initial_residual == nullptr) {
            GKO_NOT_SUPPORTED(nullptr);
        }

        auto exec = factory->get_executor();

        auto dense_r = as<Vector>(args.initial_residual);
        this->starting_tau_ = NormVector::create(
            exec, dim<2>{1, args.initial_residual->get_size()[1]});
        this->u_dense_tau_ =
            NormVector::create_with_config_of(this->starting_tau_.get());
        dense_r->compute_norm2(this->starting_tau_.get());
    }
};


/**
 * The RelativeResidualNorm class is a stopping criterion which stops the
 * iteration process when the residual norm is below a certain
 * threshold relative to the norm of the right-hand side.
 * For better performance, the checks are run thanks to kernels on
 * the executor where the algorithm is executed.
 *
 * @note To use this stopping criterion there are some dependencies. The
 * constructor depends on `b` in order to compute the norm of the
 * right-hand side. If this is not correctlyprovided, an exception
 * ::gko::NotSupported() is thrown.
 *
 * @ingroup stop
 */
template <typename ValueType = default_precision>
class RelativeResidualNorm : public ResidualNorm<ValueType> {
public:
    using NormVector = matrix::Dense<remove_complex<ValueType>>;
    using Vector = matrix::Dense<ValueType>;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Relative residual norm goal
         */
        remove_complex<ValueType> GKO_FACTORY_PARAMETER(tolerance, 1e-15);
    };
    GKO_ENABLE_CRITERION_FACTORY(RelativeResidualNorm<ValueType>, parameters,
                                 Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    explicit RelativeResidualNorm(std::shared_ptr<const gko::Executor> exec)
        : ResidualNorm<ValueType>(exec)
    {}

    explicit RelativeResidualNorm(const Factory *factory,
                                  const CriterionArgs &args)
        : ResidualNorm<ValueType>(factory->get_executor(),
                                  factory->get_parameters().tolerance),
          parameters_{factory->get_parameters()}
    {
        if (args.b == nullptr) {
            GKO_NOT_SUPPORTED(nullptr);
        }

        auto exec = factory->get_executor();

        auto dense_rhs = as<Vector>(args.b);
        this->starting_tau_ =
            NormVector::create(exec, dim<2>{1, args.b->get_size()[1]});
        this->u_dense_tau_ =
            NormVector::create_with_config_of(this->starting_tau_.get());
        dense_rhs->compute_norm2(this->starting_tau_.get());
    }
};


/**
 * The AbsoluteResidualNorm class is a stopping criterion which stops the
 * iteration process when the residual norm is below a certain
 * threshold. For better performance, the checks are run thanks to kernels on
 * the executor where the algorithm is executed.
 *
 * @note To use this stopping criterion there are some dependencies. The
 * constructor depends on `b` in order to get the number of right-hand sides.
 * If this is not correctly provided, an exception ::gko::NotSupported()
 * is thrown.
 *
 * @ingroup stop
 */
template <typename ValueType = default_precision>
class AbsoluteResidualNorm : public ResidualNorm<ValueType> {
public:
    using NormVector = matrix::Dense<remove_complex<ValueType>>;
    using Vector = matrix::Dense<ValueType>;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Absolute residual norm goal
         */
        remove_complex<ValueType> GKO_FACTORY_PARAMETER(tolerance, 1e-15);
    };
    GKO_ENABLE_CRITERION_FACTORY(AbsoluteResidualNorm<ValueType>, parameters,
                                 Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    void initialize_starting_tau();

    explicit AbsoluteResidualNorm(std::shared_ptr<const gko::Executor> exec)
        : ResidualNorm<ValueType>(exec)
    {}

    explicit AbsoluteResidualNorm(const Factory *factory,
                                  const CriterionArgs &args)
        : ResidualNorm<ValueType>(factory->get_executor(),
                                  factory->get_parameters().tolerance),
          parameters_{factory->get_parameters()}
    {
        if (args.b == nullptr) {
            GKO_NOT_SUPPORTED(nullptr);
        }

        auto exec = factory->get_executor();

        this->starting_tau_ =
            NormVector::create(exec, dim<2>{1, args.b->get_size()[1]});
        this->u_dense_tau_ =
            NormVector::create_with_config_of(this->starting_tau_.get());
        initialize_starting_tau();
    }
};


}  // namespace stop
}  // namespace gko


#endif  // GKO_CORE_STOP_RESIDUAL_NORM_HPP_
