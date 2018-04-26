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

#ifndef GKO_CORE_STOP_CRITERION_HPP_
#define GKO_CORE_STOP_CRITERION_HPP_


#include "core/base/array.hpp"
#include "core/base/lin_op.hpp"
#include "core/base/utils.hpp"


namespace gko {
namespace stop {

/**
 * The Criterion class is a base class for all stopping criterion tests. It
 * contains a factory to instantiate tests. It is up to each specific stopping
 criterion test to decide what to do with the data that is passed to it.

 * Note that depending on the tests, convergence may not have happened after
 stopping.
 */
class Criterion {
public:
    class Factory {
    public:
        /**
         * Creates the stopping criterion test.
         *
         * @param system_matrix the tested LinOp's system matrix
         * @param b the tested LinOp's input vector(s)
         * @param x the tested LinOp's output vector(s)
         *
         * @return The newly created stopping criterion test
         */
        virtual std::unique_ptr<Criterion> create_criterion(
            std::shared_ptr<const LinOp> system_matrix,
            std::shared_ptr<const LinOp> b, const LinOp *x) const = 0;
        virtual ~Factory() = default;
    };

    /**
     * The Updater class serves for pretty argument passing to the Criterion's
     * check function. The pattern used is a Builder, except Updater builds a
     * function's arguments before calling the function itself, and does not
     * build an object. This allows calling a Criterion's check in the form of:
     * stop_criterion->update()
     *   .num_iterations(num_iterations)
     *   .residual_norm(residual_norm)
     *   .residual(residual)
     *   .solution(solution)
     *   .check(converged);
     *
     * If there is a need for a new form of data to pass to the Criterion, it
     * should be added here.
     */
    class Updater {
        friend class Criterion;

    public:
        /**
         * Prevent copying and moving the object
         * This is to enforce the use of argument passing and calling check at
         * the same time.
         */
        Updater(const Updater &) = delete;
        Updater(Updater &&) = delete;
        Updater &operator=(const Updater &) = delete;
        Updater &operator=(Updater &&) = delete;

        /**
         * Calls the parent Criterion object's check method
         * @copydoc Criterion::check(Array<bool>)

         */
        bool check(Array<bool> &converged) const
        {
            return parent_->check(converged, *this);
        }

        /**
         * Helper macro to add parameters and setters to updater
         */
#define GKO_UPDATER_REGISTER_PARAMETER(_type, _name) \
    const Updater &_name(_type const &value) const   \
    {                                                \
        _name##_ = value;                            \
        return *this;                                \
    }                                                \
    mutable _type _name##_ {}

        GKO_UPDATER_REGISTER_PARAMETER(size_type, num_iterations);
        GKO_UPDATER_REGISTER_PARAMETER(const LinOp *, residual);
        GKO_UPDATER_REGISTER_PARAMETER(const LinOp *, residual_norm);
        GKO_UPDATER_REGISTER_PARAMETER(const LinOp *, solution);

#undef GKO_UPDATER_REGISTER_PARAMETER

    private:
        Updater(Criterion *parent) : parent_{parent} {}

        Criterion *parent_;
    };


    virtual ~Criterion() = default;


    Updater update() { return {this}; };

protected:
    /**
     * This checks whether convergence was reached for a certain criterion.
     * The actual implantation of the criterion goes here.
     *
     * @param converged outputs where convergence was reached
     * @param updater the Updater object containing all the information
     *
     * @returns whether convergence was completely reached
     */
    virtual bool check(Array<bool> &converged, const Updater &updater) = 0;
};


}  // namespace stop
}  // namespace gko

#endif  // GKO_CORE_STOP_CRITERION_HPP_
