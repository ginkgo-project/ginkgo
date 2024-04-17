// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_STOP_CG_ERROR_ESTIMATOR_HPP_
#define GKO_PUBLIC_CORE_STOP_CG_ERROR_ESTIMATOR_HPP_


#include <vector>


#include <ginkgo/core/stop/criterion.hpp>

namespace gko {
namespace stop {

/**
 * The CgErrorEstimator class is a stopping criterion which stops the CG process
 * by estimating the error bound.
 *
 * @note to use this stopping criterion, it is required to update the
 * alpha for the ::check() method.
 *
 * @ingroup stop
 */
template <typename ValueType = default_precision>
class CgErrorEstimator
    : public EnablePolymorphicObject<CgErrorEstimator<ValueType>, Criterion> {
    friend class EnablePolymorphicObject<CgErrorEstimator<ValueType>,
                                         Criterion>;

public:
    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * the tolerance goal
         */
        remove_complex<ValueType> GKO_FACTORY_PARAMETER_SCALAR(
            tolerance, static_cast<remove_complex<ValueType>>(1e-15));

        /**
         * tau
         */
        remove_complex<ValueType> GKO_FACTORY_PARAMETER_SCALAR(
            tau, static_cast<remove_complex<ValueType>>(0.25));
    };
    GKO_ENABLE_CRITERION_FACTORY(CgErrorEstimator<ValueType>, parameters,
                                 Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    bool check_impl(uint8 stoppingId, bool setFinalized,
                    array<stopping_status>* stop_status, bool* one_changed,
                    const Criterion::Updater& updater) override;

    explicit CgErrorEstimator(std::shared_ptr<const gko::Executor> exec)
        : EnablePolymorphicObject<CgErrorEstimator, Criterion>(std::move(exec))
    {}

    explicit CgErrorEstimator(const Factory* factory, const CriterionArgs& args)
        : EnablePolymorphicObject<CgErrorEstimator, Criterion>(
              factory->get_executor()),
          parameters_{factory->get_parameters()},
          k_{0},
          d_{0}
    {}

private:
    int k_;
    int d_;
    // TODO: should it handled through accelerator and multiple rhs?
    std::vector<ValueType> delta_;
    std::vector<ValueType> curve_;
};


}  // namespace stop
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_STOP_CG_ERROR_ESTIMATOR_HPP_
