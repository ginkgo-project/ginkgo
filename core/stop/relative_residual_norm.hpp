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

#ifndef GKO_CORE_STOP_RELATIVE_RESIDUAL_NORM_HPP_
#define GKO_CORE_STOP_RELATIVE_RESIDUAL_NORM_HPP_


#include "core/matrix/dense.hpp"
#include "core/stop/criterion.hpp"


#include <type_traits>


namespace gko {
namespace stop {

/**
 * The RelativeResidualNorm class is a stopping criterion which considers
 * convergence happened once the relative residual norm is below a certain
 * threshold. For better performance, the checks are run thanks to kernels on
 * the executor where the algorithm is executed.
 */
template <typename ValueType = default_precision>
class RelativeResidualNorm : public Criterion {
public:
    using Vector = matrix::Dense<ValueType>;

    struct Factory : public Criterion::Factory {
        /**
         * Instantiates a RelativeResidualNorm::Factory object
         * @param v the number of iterations
         * @param exec the executor to run the check on.
         */
        explicit Factory(remove_complex<ValueType> v,
                         std::shared_ptr<const gko::Executor> exec)
            : v_{v}, exec_{exec}
        {}

        static std::unique_ptr<Factory> create(
            remove_complex<ValueType> v,
            std::shared_ptr<const gko::Executor> exec)
        {
            return std::unique_ptr<Factory>(new Factory(v, exec));
        }

        std::unique_ptr<Criterion> create_criterion(
            std::shared_ptr<const LinOp> system_matrix,
            std::shared_ptr<const LinOp> b, const LinOp *x) const override;

        remove_complex<ValueType> v_;
        std::shared_ptr<const gko::Executor> exec_;
    };

    /**
     * Instantiates a RelativeResidualNorm object
     * @param v the number of iterations
     * @param exec the executor to run the kernels on
     */
    explicit RelativeResidualNorm(remove_complex<ValueType> goal,
                                  std::shared_ptr<const gko::Executor> exec)
        : rel_residual_goal_{goal}, exec_{exec}, initialized_tau_{false}
    {
        starting_tau_ = Vector::create(exec_);
    }

    bool check(uint8 stoppingId, bool setFinalized,
               Array<stopping_status> *stop_status, bool *one_changed,
               const Updater &) override;

private:
    remove_complex<ValueType> rel_residual_goal_;
    std::shared_ptr<const gko::Executor> exec_;
    std::unique_ptr<Vector> starting_tau_;
    bool initialized_tau_;
};


}  // namespace stop
}  // namespace gko


#endif  // GKO_CORE_STOP_RELATIVE_RESIDUAL_NORM_HPP_
