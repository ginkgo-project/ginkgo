/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2019

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


#include <ginkgo/core/stop/criterion.hpp>


#include <vector>


namespace gko {
namespace stop {


/**
 * The Combined class is used to combine multiple criterions together through an
 * OR operation. The typical use case is to stop the iteration process if any of
 * the criteria is fulfilled, e.g. a number of iterations, the relative residual
 * norm has reached a threshold, etc.
 */
class Combined : public EnablePolymorphicObject<Combined, Criterion> {
    friend class EnablePolymorphicObject<Combined, Criterion>;

public:
    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Criterion factories to combine
         *
         * @internal In order to simplify the factory, the factories are passed
         * as `shared_ptr`. This way there is no problem when copying the whole
         * vector. Technically it is a vector (similar to a traditional array)
         * of pointers, so copying it when creating the factories should not be
         * too costly.
         */
        std::vector<std::shared_ptr<const CriterionFactory>>
            GKO_FACTORY_PARAMETER(criteria);
    };
    GKO_ENABLE_CRITERION_FACTORY(Combined, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    bool check_impl(uint8 stoppingId, bool setFinalized,
                    Array<stopping_status> *stop_status, bool *one_changed,
                    const Updater &) override;

    explicit Combined(std::shared_ptr<const gko::Executor> exec)
        : EnablePolymorphicObject<Combined, Criterion>(std::move(exec))
    {}

    explicit Combined(const Factory *factory, const CriterionArgs &args)
        : EnablePolymorphicObject<Combined, Criterion>(factory->get_executor()),
          parameters_{factory->get_parameters()}
    {
        for (const auto &f : parameters_.criteria) {
            criteria_.push_back(f->generate(args));
        }
    }

private:
    std::vector<std::unique_ptr<Criterion>> criteria_{};
};


/**
 * Combines multiple criterion factories into a single combined criterion
 * factory.
 *
 * This function treats a singleton container as a special case and avoids
 * creating an additional object and just returns the input factory.
 *
 * @tparam FactoryContainer  a random access container type
 *
 * @param factories  a list of factories to combined
 *
 * @return a combined criterion factory if the input contains multiple factories
 *         or the input factory if the input contains only one factory
 */
template <typename FactoryContainer>
std::shared_ptr<const CriterionFactory> combine(FactoryContainer &&factories)
{
    switch (factories.size()) {
    case 0:
        GKO_NOT_SUPPORTED(nullptr);
        return nullptr;
    case 1:
        return factories[0];
    default:
        auto exec = factories[0]->get_executor();
        return Combined::build()
            .with_criteria(std::forward<FactoryContainer>(factories))
            .on(exec);
    }
}


}  // namespace stop
}  // namespace gko


#endif  // GKO_CORE_STOP_COMBINED_HPP_
