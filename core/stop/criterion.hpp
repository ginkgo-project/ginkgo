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


#include "core/base/abstract_factory.hpp"
#include "core/base/array.hpp"
#include "core/base/lin_op.hpp"
#include "core/base/polymorphic_object.hpp"
#include "core/base/utils.hpp"
#include "core/stop/stopping_status.hpp"


namespace gko {
namespace stop {


/**
 * The Criterion class is a base class for all stopping criterion tests. It
 * contains a factory to instantiate tests. It is up to each specific stopping
 * criterion test to decide what to do with the data that is passed to it.
 *
 * Note that depending on the tests, convergence may not have happened after
 * stopping.
 */
class Criterion : public EnableAbstractPolymorphicObject<Criterion> {
public:
    // class Factory {
    // public:
    //     /**
    //      * Creates the stopping criterion test.
    //      *
    //      * @param system_matrix  the tested LinOp's system matrix
    //      * @param b  the tested LinOp's input vector(s)
    //      * @param x  the tested LinOp's output vector(s)
    //      *
    //      * @return the newly created stopping criterion test
    //      */
    //     virtual std::unique_ptr<Criterion> create_criterion(
    //         std::shared_ptr<const LinOp> system_matrix,
    //         std::shared_ptr<const LinOp> b, const LinOp *x) const = 0;
    //     virtual ~Factory() = default;
    // };

    /**
     * The Updater class serves for convenient argument passing to the
     * Criterion's check function. The pattern used is a Builder, except Updater
     * builds a function's arguments before calling the function itself, and
     * does not build an object. This allows calling a Criterion's check in the
     * form of: stop_criterion->update() .num_iterations(num_iterations)
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
        bool check(uint8 stoppingId, bool setFinalized,
                   Array<stopping_status> *stop_status, bool *one_changed) const
        {
            return parent_->check(stoppingId, setFinalized, stop_status,
                                  one_changed, *this);
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


    Updater update() { return {this}; }

public:
    /**
     * This checks whether convergence was reached for a certain criterion.
     * The actual implantation of the criterion goes here.
     *
     * @param stoppingId  id of the stopping criteria
     * @param setFinalized  Controls if the current version should count as
     *                      finalized or not
     * @param stop_status  status of the stopping criteria
     * @param one_changed  indicates if one vector's status changed
     * @param updater  the Updater object containing all the information
     *
     * @returns whether convergence was completely reached
     */
    virtual bool check(uint8 stoppingId, bool setFinalized,
                       Array<stopping_status> *stop_status, bool *one_changed,
                       const Updater &updater) = 0;

protected:
    explicit Criterion(std::shared_ptr<const gko::Executor> exec)
        : EnableAbstractPolymorphicObject<Criterion>(exec)
    {}
};


struct CriterionArgs {
    std::shared_ptr<const LinOp> system_matrix;
    std::shared_ptr<const LinOp> b;
    const LinOp *x;

    explicit CriterionArgs(std::shared_ptr<const LinOp> system_matrix,
                           std::shared_ptr<const LinOp> b, const LinOp *x)
    {
        this->system_matrix = std::move(system_matrix);
        this->b = std::move(b);
        this->x = x;
    }
};

/**
 * Declares an Abstract Factory specialized for Criterions
 */
using CriterionFactory = AbstractFactory<Criterion, const CriterionArgs *>;


/**
 * This is an alias for the EnableDefaultFactory mixin, which correctly sets the
 * template parameters to enable a subclass of CriterionFactory.
 *
 * @tparam ConcreteFactory  the concrete factory which is being implemented
 *                          [CRTP parmeter]
 * @tparam ConcreteLinOp  the concrete LinOp type which this factory produces,
 *                        needs to have a constructor which takes a
 *                        const ConcreteFactory *, and an
 *                        std::shared_ptr<const LinOp> as parameters.
 * @tparam ParametersType  a subclass of enable_parameters_type template which
 *                         defines all of the parameters of the factory
 * @tparam PolymorphicBase  parent of ConcreteFactory in the polymorphic
 *                          hierarchy, has to be a subclass of CriterionFactory
 */
template <typename ConcreteFactory, typename ConcreteCriterion,
          typename ParametersType, typename PolymorphicBase = CriterionFactory>
using EnableDefaultCriterionFactory =
    EnableDefaultFactory<ConcreteFactory, ConcreteCriterion, ParametersType,
                         PolymorphicBase>;


#define GKO_CREATE_CRITERION_PARAMETERS(_parameters_name, _factory_name) \
    class _factory_name;                                                 \
                                                                         \
public:                                                                  \
    struct _parameters_name##_type                                       \
        : ::gko::enable_parameters_type<_parameters_name##_type,         \
                                        _factory_name>

/**
 * This macro will generate a default implementation of a CriterionFactory for
 * the Criterion subclass it is defined in.
 *
 * This macro is very similar to the macro #ENABLE_LIN_OP_FACTORY(). A more
 * detailed description of the use of these type of macros can be found there.
 *
 * @param _criterion  concrete operator for which the factory is to be created
 *                    [CRTP parameter]
 * @param _parameters_name  name of the parameters member in the class
 *                          (its type is `<_parameters_name>_type`, the
 *                          protected member's name is `<_parameters_name>_`,
 *                          and the public getter's name is
 *                          `get_<_parameters_name>()`)
 * @param _factory_name  name of the generated factory type
 */
#define GKO_ENABLE_CRITERION_FACTORY(_criterion, _parameters_name,          \
                                     _factory_name)                         \
public:                                                                     \
    const _parameters_name##_type &get_##_parameters_name() const           \
    {                                                                       \
        return _parameters_name##_;                                         \
    }                                                                       \
                                                                            \
    class _factory_name                                                     \
        : public ::gko::stop::EnableDefaultCriterionFactory<                \
              _factory_name, _criterion, _parameters_name##_type> {         \
        friend class ::gko::EnablePolymorphicObject<                        \
            _factory_name, ::gko::stop::CriterionFactory>;                  \
        friend class ::gko::enable_parameters_type<_parameters_name##_type, \
                                                   _factory_name>;          \
        using ::gko::stop::EnableDefaultCriterionFactory<                   \
            _factory_name, _criterion,                                      \
            _parameters_name##_type>::EnableDefaultCriterionFactory;        \
    };                                                                      \
    friend ::gko::stop::EnableDefaultCriterionFactory<                      \
        _factory_name, _criterion, _parameters_name##_type>;                \
                                                                            \
private:                                                                    \
    _parameters_name##_type _parameters_name##_


}  // namespace stop
}  // namespace gko

#endif  // GKO_CORE_STOP_CRITERION_HPP_
