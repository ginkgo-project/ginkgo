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

#ifndef GKO_CORE_STOP_COMBINED_HPP_
#define GKO_CORE_STOP_COMBINED_HPP_


#include "core/stop/criterion.hpp"


#include <vector>


namespace gko {
namespace stop {


/**
 * The Combined class is used to combine multiple criterions together through an
 * OR operation. The typical use case is to define convergence if any of the
 * criteria is fulfilled, e.g. a number of iterations, the relative residual
 * norm has reached a threshold, etc.
 */
class Combined : public EnablePolymorphicObject<Combined, Criterion> {
    friend class EnablePolymorphicObject<Combined, Criterion>;

public:
    bool check(uint8 stoppingId, bool setFinalized,
               Array<stopping_status> *stop_status, bool *one_changed,
               const Updater &) override;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Criterion factories to combine
         */
        std::vector<std::shared_ptr<const CriterionFactory>>
            GKO_FACTORY_PARAMETER(criteria, );
    };
    GKO_ENABLE_CRITERION_FACTORY(Combined, parameters, Factory);

protected:
    explicit Combined(std::shared_ptr<const gko::Executor> exec)
        : EnablePolymorphicObject<Combined, Criterion>(std::move(exec))
    {}

    explicit Combined(const Factory *factory, const CriterionArgs *args)
        : EnablePolymorphicObject<Combined, Criterion>(factory->get_executor()),
          parameters_{factory->get_parameters()}
    {
        for (const auto &f : parameters_.criteria)
            criteria_.push_back(std::move(f->generate(args)));
    }

private:
    std::vector<std::unique_ptr<Criterion>> criteria_{};
};


}  // namespace stop
}  // namespace gko


#endif  // GKO_CORE_STOP_COMBINED_HPP_
