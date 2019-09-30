/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_REORDER_REORDERING_BASE_HPP_
#define GKO_CORE_REORDER_REORDERING_BASE_HPP_


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
class ReorderingBase : public EnableAbstractPolymorphicObject<ReorderingBase> {
protected:
    explicit ReorderingBase(std::shared_ptr<const gko::Executor> exec)
        : EnableAbstractPolymorphicObject<ReorderingBase>(exec)
    {}
};


/**
 * This struct is used to pass parameters to the
 * EnableDefaultReorderingBaseFactory::generate() method. It is the
 * ComponentsType of ReorderingBaseFactory.
 */
struct ReorderingBaseArgs {
    std::shared_ptr<LinOp> adjacency_matrix;

    ReorderingBaseArgs(std::shared_ptr<LinOp> adjacency_matrix)
        : adjacency_matrix{adjacency_matrix}
    {}
};


/**
 * Declares an Abstract Factory specialized for ReorderingBases
 */
using ReorderingBaseFactory =
    AbstractFactory<ReorderingBase, ReorderingBaseArgs>;


/**
 * This is an alias for the EnableDefaultFactory mixin, which correctly sets the
 * template parameters to enable a subclass of ReorderingBaseFactory.
 *
 * @tparam ConcreteFactory  the concrete factory which is being implemented
 *                          [CRTP parmeter]
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
          typename ParametersType,
          typename PolymorphicBase = ReorderingBaseFactory>
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
    const _parameters_name##_type &get_##_parameters_name() const              \
    {                                                                          \
        return _parameters_name##_;                                            \
    }                                                                          \
                                                                               \
    class _factory_name                                                        \
        : public ::gko::reorder::EnableDefaultReorderingBaseFactory<           \
              _factory_name, _reordering_base, _parameters_name##_type> {      \
        friend class ::gko::EnablePolymorphicObject<                           \
            _factory_name, ::gko::reorder::ReorderingBaseFactory>;             \
        friend class ::gko::enable_parameters_type<_parameters_name##_type,    \
                                                   _factory_name>;             \
        using ::gko::reorder::EnableDefaultReorderingBaseFactory<              \
            _factory_name, _reordering_base,                                   \
            _parameters_name##_type>::EnableDefaultReorderingBaseFactory;      \
    };                                                                         \
    friend ::gko::reorder::EnableDefaultReorderingBaseFactory<                 \
        _factory_name, _reordering_base, _parameters_name##_type>;             \
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


#endif  // GKO_CORE_REORDER_REORDERING_BASE_HPP_
