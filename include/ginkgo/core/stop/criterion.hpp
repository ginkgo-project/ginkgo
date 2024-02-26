// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_STOP_CRITERION_HPP_
#define GKO_PUBLIC_CORE_STOP_CRITERION_HPP_


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>


namespace gko {
/**
 * @brief The Stopping criterion namespace.
 *
 * @ingroup stop
 */
namespace stop {


/**
 * The Criterion class is a base class for all stopping criteria. It
 * contains a factory to instantiate criteria. It is up to each specific
 * stopping criterion to decide what to do with the data that is passed to it.
 *
 * Note that depending on the criterion, convergence may not have happened after
 * stopping.
 */
class Criterion : public EnableAbstractPolymorphicObject<Criterion> {
public:
    /**
     * The Updater class serves for convenient argument passing to the
     * Criterion's check function. The pattern used is a Builder, except Updater
     * builds a function's arguments before calling the function itself, and
     * does not build an object. This allows calling a Criterion's check in the
     * form of: stop_criterion->update()
     *   .num_iterations(num_iterations)
     *   .ignore_residual_check(ignore_residual_check)
     *   .residual_norm(residual_norm)
     *   .implicit_sq_residual_norm(implicit_sq_residual_norm)
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
        Updater(const Updater&) = delete;
        Updater(Updater&&) = delete;
        Updater& operator=(const Updater&) = delete;
        Updater& operator=(Updater&&) = delete;

        /**
         * Calls the parent Criterion object's check method
         * @copydoc Criterion::check(uint8, bool, array<stopping_status>, bool)
         */
        bool check(uint8 stopping_id, bool set_finalized,
                   array<stopping_status>* stop_status, bool* one_changed) const
        {
            auto converged = parent_->check(stopping_id, set_finalized,
                                            stop_status, one_changed, *this);
            return converged;
        }

        /**
         * Helper macro to add parameters and setters to updater
         */
#define GKO_UPDATER_REGISTER_PARAMETER(_type, _name) \
    const Updater& _name(_type const& value) const   \
    {                                                \
        _name##_ = value;                            \
        return *this;                                \
    }                                                \
    mutable _type _name##_ {}
#define GKO_UPDATER_REGISTER_PTR_PARAMETER(_type, _name) \
    const Updater& _name(ptr_param<_type> value) const   \
    {                                                    \
        _name##_ = value.get();                          \
        return *this;                                    \
    }                                                    \
    mutable _type* _name##_ {}

        GKO_UPDATER_REGISTER_PARAMETER(size_type, num_iterations);
        // ignore_residual_check default is false
        GKO_UPDATER_REGISTER_PARAMETER(bool, ignore_residual_check);
        GKO_UPDATER_REGISTER_PTR_PARAMETER(const LinOp, residual);
        GKO_UPDATER_REGISTER_PTR_PARAMETER(const LinOp, residual_norm);
        GKO_UPDATER_REGISTER_PTR_PARAMETER(const LinOp,
                                           implicit_sq_residual_norm);
        GKO_UPDATER_REGISTER_PTR_PARAMETER(const LinOp, solution);

#undef GKO_UPDATER_REGISTER_PTR_PARAMETER
#undef GKO_UPDATER_REGISTER_PARAMETER

    private:
        Updater(Criterion* parent) : parent_{parent} {}

        Criterion* parent_;
    };

    /**
     * Returns the updater object
     *
     * @return the updater object
     */
    Updater update() { return {this}; }

    /**
     * This checks whether convergence was reached for a certain criterion.
     * The actual implantation of the criterion goes here.
     *
     * @param stopping_id  id of the stopping criterion
     * @param set_finalized  Controls if the current version should count as
     *                      finalized or not
     * @param stop_status  status of the stopping criterion
     * @param one_changed  indicates if the status of a vector has changed
     * @param updater  the Updater object containing all the information
     *
     * @returns whether convergence was completely reached
     */
    bool check(uint8 stopping_id, bool set_finalized,
               array<stopping_status>* stop_status, bool* one_changed,
               const Updater& updater)
    {
        this->template log<log::Logger::criterion_check_started>(
            this, updater.num_iterations_, updater.residual_,
            updater.residual_norm_, updater.solution_, stopping_id,
            set_finalized);
        auto all_converged = this->check_impl(
            stopping_id, set_finalized, stop_status, one_changed, updater);
        this->template log<log::Logger::criterion_check_completed>(
            this, updater.num_iterations_, updater.residual_,
            updater.residual_norm_, updater.implicit_sq_residual_norm_,
            updater.solution_, stopping_id, set_finalized, stop_status,
            *one_changed, all_converged);
        return all_converged;
    }

protected:
    /**
     * Implementers of Criterion should override this function instead
     * of check(uint8, bool, array<stopping_status>*, bool*, const Updater&).
     *
     * This checks whether convergence was reached for a certain criterion.
     * The actual implantation of the criterion goes here.
     *
     * @param stopping_id  id of the stopping criterion
     * @param set_finalized  Controls if the current version should count as
     *                      finalized or not
     * @param stop_status  status of the stopping criterion
     * @param one_changed  indicates if the status of a vector has changed
     * @param updater  the Updater object containing all the information
     *
     * @returns whether convergence was completely reached
     */
    virtual bool check_impl(uint8 stopping_id, bool set_finalized,
                            array<stopping_status>* stop_status,
                            bool* one_changed, const Updater& updater) = 0;

    /**
     * This is a helper function which properly sets all elements of the
     * stopping_status to converged. This is used in stopping criteria such as
     * Time or Iteration.
     *
     * @param stopping_id  id of the stopping criterion
     * @param set_finalized  Controls if the current version should count as
     *                      finalized or not
     * @param stop_status  status of the stopping criterion
     */
    void set_all_statuses(uint8 stopping_id, bool set_finalized,
                          array<stopping_status>* stop_status);

    explicit Criterion(std::shared_ptr<const gko::Executor> exec)
        : EnableAbstractPolymorphicObject<Criterion>(exec)
    {}
};


/**
 * This struct is used to pass parameters to the
 * EnableDefaultCriterionFactoryCriterionFactory::generate() method. It is the
 * ComponentsType of CriterionFactory.
 *
 * @note Dependly on the use case, some of these parameters can be `nullptr` as
 * only some stopping criterion require them to be set. An example is the
 * `ResidualNormReduction` which really requires the `initial_residual` to be
 * set.
 */
struct CriterionArgs {
    std::shared_ptr<const LinOp> system_matrix;
    std::shared_ptr<const LinOp> b;
    const LinOp* x;
    const LinOp* initial_residual;


    CriterionArgs(std::shared_ptr<const LinOp> system_matrix,
                  std::shared_ptr<const LinOp> b, const LinOp* x,
                  const LinOp* initial_residual = nullptr)
        : system_matrix{system_matrix},
          b{b},
          x{x},
          initial_residual{initial_residual}
    {}
};


/**
 * Declares an Abstract Factory specialized for Criterions
 */
using CriterionFactory = AbstractFactory<Criterion, CriterionArgs>;


/**
 * This is an alias for the EnableDefaultFactory mixin, which correctly sets the
 * template parameters to enable a subclass of CriterionFactory.
 *
 * @tparam ConcreteFactory  the concrete factory which is being implemented
 *                          [CRTP parameter]
 * @tparam ConcreteCriterion  the concrete Criterion type which this factory
 *                            produces, needs to have a constructor which takes
 *                            a const ConcreteFactory *, and a
 *                            const CriterionArgs * as parameters.
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
 *
 * @internal For some abstract reason, nvcc compilation through HIP does not
 *           properly take into account the `using` declaration to inherit
 *           constructors. In addition, the default initialization `{}` for
 *           `_parameters_name##type parameters` also does not work, which
 *           means the current form is probably the only correct one.
 *
 * @ingroup stop
 */
#define GKO_ENABLE_CRITERION_FACTORY(_criterion, _parameters_name,           \
                                     _factory_name)                          \
public:                                                                      \
    const _parameters_name##_type& get_##_parameters_name() const            \
    {                                                                        \
        return _parameters_name##_;                                          \
    }                                                                        \
                                                                             \
    class _factory_name                                                      \
        : public ::gko::stop::EnableDefaultCriterionFactory<                 \
              _factory_name, _criterion, _parameters_name##_type> {          \
        friend class ::gko::EnablePolymorphicObject<                         \
            _factory_name, ::gko::stop::CriterionFactory>;                   \
        friend class ::gko::enable_parameters_type<_parameters_name##_type,  \
                                                   _factory_name>;           \
        explicit _factory_name(std::shared_ptr<const ::gko::Executor> exec)  \
            : ::gko::stop::EnableDefaultCriterionFactory<                    \
                  _factory_name, _criterion, _parameters_name##_type>(       \
                  std::move(exec))                                           \
        {}                                                                   \
        explicit _factory_name(std::shared_ptr<const ::gko::Executor> exec,  \
                               const _parameters_name##_type& parameters)    \
            : ::gko::stop::EnableDefaultCriterionFactory<                    \
                  _factory_name, _criterion, _parameters_name##_type>(       \
                  std::move(exec), parameters)                               \
        {}                                                                   \
    };                                                                       \
    friend ::gko::stop::EnableDefaultCriterionFactory<                       \
        _factory_name, _criterion, _parameters_name##_type>;                 \
                                                                             \
private:                                                                     \
    _parameters_name##_type _parameters_name##_;                             \
                                                                             \
public:                                                                      \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


}  // namespace stop
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_STOP_CRITERION_HPP_
