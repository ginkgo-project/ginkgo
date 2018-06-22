/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_STOP_RESIDUAL_NORM_REDUCTION_HPP_
#define GKO_CORE_STOP_RESIDUAL_NORM_REDUCTION_HPP_


#include "core/matrix/dense.hpp"
#include "core/stop/criterion.hpp"


#include <type_traits>


namespace gko {
namespace stop {

/**
 * The ResidualNormReduction class is a stopping criterion which stops the
 * iteration process when the relative residual norm is below a certain
 * threshold. For better performance, the checks are run thanks to kernels on
 * the executor where the algorithm is executed.
 *
 * @note To use this stopping criterion there are some dependencies on: `b` in
 * order to know the size of the tau vector, either the `residual_norm` or the
 * `residual`. When any of those is not correctly provided, an exception
 * ::gko::NotImplemented() is thrown.
 */
template <typename ValueType = default_precision>
class ResidualNormReduction
    : public EnablePolymorphicObject<ResidualNormReduction<ValueType>,
                                     Criterion> {
    friend class EnablePolymorphicObject<ResidualNormReduction<ValueType>,
                                         Criterion>;

public:
    using Vector = matrix::Dense<ValueType>;

    bool check(uint8 stoppingId, bool setFinalized,
               Array<stopping_status> *stop_status, bool *one_changed,
               const Criterion::Updater &) override;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Relative residual norm goal
         */
        remove_complex<ValueType> GKO_FACTORY_PARAMETER(reduction_factor,
                                                        1e-15);
    };
    GKO_ENABLE_CRITERION_FACTORY(ResidualNormReduction<ValueType>, parameters,
                                 Factory);

protected:
    explicit ResidualNormReduction(std::shared_ptr<const gko::Executor> exec)
        : EnablePolymorphicObject<ResidualNormReduction<ValueType>, Criterion>(
              std::move(exec))
    {}

    explicit ResidualNormReduction(const Factory *factory,
                                   const CriterionArgs args)
        : EnablePolymorphicObject<ResidualNormReduction<ValueType>, Criterion>(
              factory->get_executor()),
          parameters_{factory->get_parameters()}
    {
        if (args.b == nullptr) {
            NOT_IMPLEMENTED;
        }
        starting_tau_ = Vector::create(factory->get_executor(),
                                       dim{1, args.b->get_size().num_cols});
    }

private:
    std::unique_ptr<Vector> starting_tau_{};
    bool initialized_tau_{};
};


}  // namespace stop
}  // namespace gko


#endif  // GKO_CORE_STOP_RESIDUAL_NORM_REDUCTION_HPP_
