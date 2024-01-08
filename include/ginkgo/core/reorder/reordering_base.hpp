// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_REORDER_REORDERING_BASE_HPP_
#define GKO_PUBLIC_CORE_REORDER_REORDERING_BASE_HPP_


#include <memory>


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/utils.hpp>


namespace gko {
/**
 * @brief The reordering namespace.
 *
 * @ingroup reorder
 */
namespace reorder {


/**
 * The ReorderingBase class is a base class for all the reordering algorithms.
 * It contains a factory to instantiate the reorderings. It is up to each
 * specific reordering to decide what to do with the data that is passed to it.
 */
template <typename IndexType = int32>
class ReorderingBase
    : public EnableAbstractPolymorphicObject<ReorderingBase<IndexType>> {
public:
    using index_type = IndexType;

    const array<index_type>& get_permutation_array() const
    {
        return permutation_array_;
    }

protected:
    explicit ReorderingBase(std::shared_ptr<const gko::Executor> exec)
        : EnableAbstractPolymorphicObject<ReorderingBase>(exec),
          permutation_array_{exec}
    {}

    void set_permutation_array(array<index_type>& permutation_array)
    {
        permutation_array_ = permutation_array;
    }

private:
    array<index_type> permutation_array_;
};


/**
 * This struct is used to pass parameters to the
 * EnableDefaultReorderingBaseFactory::generate() method. It is the
 * ComponentsType of ReorderingBaseFactory.
 */
struct ReorderingBaseArgs {
    std::shared_ptr<LinOp> system_matrix;

    ReorderingBaseArgs(std::shared_ptr<LinOp> system_matrix)
        : system_matrix{system_matrix}
    {}
};


/**
 * Declares an Abstract Factory specialized for ReorderingBases
 */
template <typename IndexType = int32>
using ReorderingBaseFactory =
    AbstractFactory<ReorderingBase<IndexType>, ReorderingBaseArgs>;


/**
 * This is an alias for the EnableDefaultFactory mixin, which correctly sets the
 * template parameters to enable a subclass of ReorderingBaseFactory.
 *
 * @tparam ConcreteFactory  the concrete factory which is being implemented
 *                          [CRTP parameter]
 * @tparam ConcreteReorderingBase  the concrete ReorderingBase type which this
 * factory produces, needs to have a constructor which takes a const
 * ConcreteFactory *, and a const ReorderingBaseArgs * as parameters.
 * @tparam ParametersType  a subclass of enable_parameters_type template which
 *                         defines all of the parameters of the factory
 * @tparam PolymorphicBase  parent of ConcreteFactory in the polymorphic
 *                          hierarchy, has to be a subclass of
 * ReorderingBaseFactory
 */
template <typename ConcreteFactory, typename ConcreteReorderingBase,
          typename ParametersType, typename IndexType = int32,
          typename PolymorphicBase = ReorderingBaseFactory<IndexType>>
using EnableDefaultReorderingBaseFactory =
    EnableDefaultFactory<ConcreteFactory, ConcreteReorderingBase,
                         ParametersType, PolymorphicBase>;


/**
 * This macro will generate a default implementation of a ReorderingBaseFactory
 * for the ReorderingBase subclass it is defined in.
 *
 * This macro is very similar to the macro #ENABLE_LIN_OP_FACTORY(). A more
 * detailed description of the use of these type of macros can be found there.
 *
 * @param _reordering_base  concrete operator for which the factory is to be
 * created [CRTP parameter]
 * @param _parameters_name  name of the parameters member in the class
 *                          (its type is `<_parameters_name>_type`, the
 *                          protected member's name is `<_parameters_name>_`,
 *                          and the public getter's name is
 *                          `get_<_parameters_name>()`)
 * @param _factory_name  name of the generated factory type
 *
 * @ingroup reorder
 */
#define GKO_ENABLE_REORDERING_BASE_FACTORY(_reordering_base, _parameters_name, \
                                           _factory_name)                      \
public:                                                                        \
    const _parameters_name##_type& get_##_parameters_name() const              \
    {                                                                          \
        return _parameters_name##_;                                            \
    }                                                                          \
                                                                               \
    class _factory_name                                                        \
        : public ::gko::reorder::EnableDefaultReorderingBaseFactory<           \
              _factory_name, _reordering_base, _parameters_name##_type,        \
              IndexType> {                                                     \
        friend class ::gko::EnablePolymorphicObject<                           \
            _factory_name, ::gko::reorder::ReorderingBaseFactory<IndexType>>;  \
        friend class ::gko::enable_parameters_type<_parameters_name##_type,    \
                                                   _factory_name>;             \
        explicit _factory_name(std::shared_ptr<const ::gko::Executor> exec)    \
            : ::gko::reorder::EnableDefaultReorderingBaseFactory<              \
                  _factory_name, _reordering_base, _parameters_name##_type,    \
                  IndexType>(std::move(exec))                                  \
        {}                                                                     \
        explicit _factory_name(std::shared_ptr<const ::gko::Executor> exec,    \
                               const _parameters_name##_type& parameters)      \
            : ::gko::reorder::EnableDefaultReorderingBaseFactory<              \
                  _factory_name, _reordering_base, _parameters_name##_type,    \
                  IndexType>(std::move(exec), parameters)                      \
        {}                                                                     \
    };                                                                         \
    friend ::gko::reorder::EnableDefaultReorderingBaseFactory<                 \
        _factory_name, _reordering_base, _parameters_name##_type, IndexType>;  \
                                                                               \
private:                                                                       \
    _parameters_name##_type _parameters_name##_;                               \
                                                                               \
public:                                                                        \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")


}  // namespace reorder
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_REORDER_REORDERING_BASE_HPP_
