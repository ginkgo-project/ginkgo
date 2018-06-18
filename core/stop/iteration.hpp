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

#ifndef GKO_CORE_STOP_ITERATION_HPP_
#define GKO_CORE_STOP_ITERATION_HPP_


#include "core/stop/criterion.hpp"


namespace gko {
namespace stop {

/**
 * The Iteration class is a stopping criterion which stops the iteration process
 * after a preset number of iterations.
 */
class Iteration : public EnablePolymorphicObject<Iteration, Criterion> {
    friend class EnablePolymorphicObject<Iteration, Criterion>;

public:
    bool check(uint8 stoppingId, bool setFinalized,
               Array<stopping_status> *stop_status, bool *one_changed,
               const Updater &updater) override;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Maximum number of iterations
         */
        size_type GKO_FACTORY_PARAMETER(max_iters, 0);
    };
    GKO_ENABLE_CRITERION_FACTORY(Iteration, parameters, Factory);

protected:
    explicit Iteration(std::shared_ptr<const gko::Executor> exec)
        : EnablePolymorphicObject<Iteration, Criterion>(std::move(exec))
    {}

    explicit Iteration(const Factory *factory, const CriterionArgs *args)
        : EnablePolymorphicObject<Iteration, Criterion>(
              factory->get_executor()),
          parameters_{factory->get_parameters()}
    {}
};


}  // namespace stop
}  // namespace gko


#endif  // GKO_CORE_STOP_ITERATION_HPP_
