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

#ifndef GKO_EXAMPLE_BYINTERACTION_HPP_
#define GKO_EXAMPLE_BYINTERACTION_HPP_


#include <core/stop/criterion.hpp>


namespace gko {
namespace stop {


#define GKO_TMP_FACTORY_PARAMETER(_name, ...)                \
    _name{__VA_ARGS__};                                      \
                                                             \
    template <typename... Args>                              \
    auto with_##_name(Args &&... _value)                     \
        const->const ::gko::xstd::decay_t<decltype(*this)> & \
    {                                                        \
        using type = decltype(this->_name);                  \
        this->_name = type{std::forward<Args>(_value)...};   \
        return *this;                                        \
    }


/**
 * The ByInteraction class is a criterion which asks for user input to stop
 * the iteration process. Using this criterion is slightly more complex than the
 * other ones, because it is asynchronous therefore requires the use of threads.
 * The following shows an usage example:
 *
 * ```C++
 * void run_solver(volatile bool &is_user_bored)
 * {
 *   using mtx = gko::matrix::Dense<>;
 *   auto exec = gko::GpuExecutor::create(gko::OmpExecutor::create(), 0);
 *   auto A = gko::read<mtx>(exec, "A.mtx");
 *   auto b = gko::read<mtx>(exec, "b.mtx");
 *   auto x = gko::read<mtx>(exec, "x0.mtx");
 *   gko::solver::CgFactory::create(exec,
 *   ByInteraction::Factory::create(exec, is_user_bored))
 *     ->generate(gko::give(A))
 *     ->apply(gko::lend(b), gko::lend(x));
 *   std::cout << "Solver stopped" << std::endl;
 * }

 * int main()
 * {
 *   volatile bool is_user_bored;
 *   std::thread t(run_solver, std::ref(is_user_bored));
 *   std::string command;
 *   while (std::cin >> command) {
 *     if (command == "stop") {
 *       break;
 *     } else {
 *       std::cout << "Unknown command" << std::endl;
 *     }
 *   }
 *   std::cout << "I see you've had enough - I'm stopping the solver!" <<
 std::endl;
 *   is_user_bored = true;
 *   t.join();
 * }
 * ```
 */
class ByInteraction : public EnablePolymorphicObject<ByInteraction, Criterion> {
    friend class EnablePolymorphicObject<ByInteraction, Criterion>;

public:
    bool check(uint8 stoppingId, bool setFinalized,
               Array<stopping_status> *stop_status, bool *one_changed,
               const Updater &) override;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Boolean set by the user to stop the iteration process
         */
        mutable volatile bool *GKO_TMP_FACTORY_PARAMETER(stop_iteration_process,
                                                         nullptr);
#undef GKO_TMP_FACTORY_PARAMETER
    };
    GKO_ENABLE_CRITERION_FACTORY(ByInteraction, parameters, Factory);

protected:
    explicit ByInteraction(std::shared_ptr<const gko::Executor> exec)
        : EnablePolymorphicObject<ByInteraction, Criterion>(std::move(exec))
    {}

    explicit ByInteraction(const Factory *factory, const CriterionArgs args)

        : EnablePolymorphicObject<ByInteraction, Criterion>(
              factory->get_executor()),
          parameters_{factory->get_parameters()}
    {}

    ByInteraction &operator=(const ByInteraction &other) { return *this; }

    ByInteraction &operator=(ByInteraction &other) { return *this; }
};


}  // namespace stop
}  // namespace gko


#endif  // GKO_EXAMPLE_BYINTERACTION_HPP_
