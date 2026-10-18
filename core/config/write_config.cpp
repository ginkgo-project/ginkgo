// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/config/write_config.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/config/property_tree.hpp>
#include <ginkgo/core/factorization/ic.hpp>
#include <ginkgo/core/factorization/ilu.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/preconditioner/ic.hpp>
#include <ginkgo/core/preconditioner/ilu.hpp>
#include <ginkgo/core/preconditioner/isai.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/triangular.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/criterion.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/core/stop/time.hpp>
#include <ginkgo/core/synthesizer/containers.hpp>

#include "core/config/dispatch.hpp"
#include "core/config/write_helper.hpp"


namespace gko {
namespace config {
namespace {


/**
 * Writes the parameters of factory to result if it is a T::Factory.
 */
template <typename T, typename AbstractFactoryType>
bool try_write(const AbstractFactoryType& factory, pnode& result)
{
    if (auto concrete = dynamic_cast<const typename T::Factory*>(&factory)) {
        result = parameter_writer<T>::write(concrete->get_parameters());
        return true;
    }
    return false;
}


template <template <typename...> class Base, typename AbstractFactoryType,
          typename... ValueTypes>
bool try_write_value_type(const AbstractFactoryType& factory, pnode& result,
                          syn::type_list<ValueTypes...>)
{
    return (try_write<Base<ValueTypes>>(factory, result) || ...);
}


template <template <typename...> class Base, typename ValueType,
          typename AbstractFactoryType, typename... IndexTypes>
bool try_write_index_type(const AbstractFactoryType& factory, pnode& result,
                          syn::type_list<IndexTypes...>)
{
    return (try_write<Base<ValueType, IndexTypes>>(factory, result) || ...);
}


template <template <typename...> class Base, typename AbstractFactoryType,
          typename... ValueTypes, typename IndexTypeList>
bool try_write_value_and_index_type(const AbstractFactoryType& factory,
                                    pnode& result,
                                    syn::type_list<ValueTypes...>,
                                    IndexTypeList index_types)
{
    return (
        try_write_index_type<Base, ValueTypes>(factory, result, index_types) ||
        ...);
}


// the type lists that the parsers dispatch over
template <template <typename...> class Base, typename AbstractFactoryType>
bool try_write_value_type(const AbstractFactoryType& factory, pnode& result)
{
    return try_write_value_type<Base>(factory, result, value_type_list());
}


template <template <typename...> class Base, typename AbstractFactoryType>
bool try_write_value_and_index_type(const AbstractFactoryType& factory,
                                    pnode& result)
{
    return try_write_value_and_index_type<Base>(
        factory, result, value_type_list(), index_type_list());
}


template <typename... ValueTypes>
bool is_identity_factory_of(const LinOpFactory& factory,
                            syn::type_list<ValueTypes...>)
{
    return ((dynamic_cast<const matrix::IdentityFactory<ValueTypes>*>(
                 &factory) != nullptr) ||
            ...);
}


}  // namespace


// IdentityFactory is instantiated with GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE
bool is_identity_factory(const LinOpFactory& factory)
{
    return is_identity_factory_of(factory, value_type_list());
}


pnode write_config(const LinOpFactory& factory)
{
    pnode result;
    if (try_write_value_type<solver::Cg>(factory, result) ||
        try_write_value_type<solver::Bicgstab>(factory, result) ||
        try_write_value_type<solver::Gmres>(factory, result) ||
        try_write_value_type<solver::Ir>(factory, result) ||
        try_write_value_and_index_type<solver::LowerTrs>(factory, result) ||
        try_write_value_and_index_type<solver::UpperTrs>(factory, result) ||
        try_write_value_and_index_type<preconditioner::Jacobi>(factory,
                                                               result) ||
        try_write_value_and_index_type<forward_ilu>(factory, result) ||
        try_write_value_and_index_type<reverse_ilu>(factory, result) ||
        try_write_value_and_index_type<preconditioner::Ic>(factory, result) ||
        try_write_value_and_index_type<preconditioner::LowerIsai>(factory,
                                                                  result) ||
        try_write_value_and_index_type<preconditioner::UpperIsai>(factory,
                                                                  result) ||
        try_write_value_and_index_type<preconditioner::GeneralIsai>(factory,
                                                                    result) ||
        try_write_value_and_index_type<preconditioner::SpdIsai>(factory,
                                                                result) ||
        try_write_value_and_index_type<factorization::Ilu>(factory, result) ||
        try_write_value_and_index_type<factorization::Ic>(factory, result)) {
        return result;
    }
    GKO_NOT_SUPPORTED(factory);
}


pnode write_config(const stop::CriterionFactory& factory)
{
    pnode result;
    if (try_write<stop::Combined>(factory, result) ||
        try_write<stop::Iteration>(factory, result) ||
        try_write<stop::Time>(factory, result) ||
        try_write_value_type<stop::ResidualNorm>(factory, result) ||
        try_write_value_type<stop::ImplicitResidualNorm>(factory, result)) {
        return result;
    }
    GKO_NOT_SUPPORTED(factory);
}


}  // namespace config
}  // namespace gko
