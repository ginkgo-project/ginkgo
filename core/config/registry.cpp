// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/config/registry.hpp>


#include "core/config/config_helper.hpp"


namespace gko {
namespace config {


template <>
linop_map& registry::get_map_impl<linop_map>()
{
    return linop_map_;
}

template <>
linopfactory_map& registry::get_map_impl<linopfactory_map>()
{
    return linopfactory_map_;
}

template <>
criterionfactory_map& registry::get_map_impl<criterionfactory_map>()
{
    return criterionfactory_map_;
}

template <>
const linop_map& registry::get_map_impl<linop_map>() const
{
    return linop_map_;
}

template <>
const linopfactory_map& registry::get_map_impl<linopfactory_map>() const
{
    return linopfactory_map_;
}

template <>
const criterionfactory_map& registry::get_map_impl<criterionfactory_map>() const
{
    return criterionfactory_map_;
}


template <typename ValueType, typename IndexType>
type_descriptor make_type_descriptor()
{
    return type_descriptor{type_string<ValueType>::str(),
                           type_string<IndexType>::str()};
}

template type_descriptor make_type_descriptor<void, void>();
template type_descriptor make_type_descriptor<float, void>();
template type_descriptor make_type_descriptor<double, void>();
template type_descriptor make_type_descriptor<std::complex<float>, void>();
template type_descriptor make_type_descriptor<std::complex<double>, void>();
template type_descriptor make_type_descriptor<void, int32>();
template type_descriptor make_type_descriptor<float, int32>();
template type_descriptor make_type_descriptor<double, int32>();
template type_descriptor make_type_descriptor<std::complex<float>, int32>();
template type_descriptor make_type_descriptor<std::complex<double>, int32>();
template type_descriptor make_type_descriptor<void, int64>();
template type_descriptor make_type_descriptor<float, int64>();
template type_descriptor make_type_descriptor<double, int64>();
template type_descriptor make_type_descriptor<std::complex<float>, int64>();
template type_descriptor make_type_descriptor<std::complex<double>, int64>();


type_descriptor::type_descriptor(std::string value_typestr,
                                 std::string index_typestr)
    : value_typestr_(value_typestr), index_typestr_(index_typestr)
{}

const std::string& type_descriptor::get_value_typestr() const
{
    return value_typestr_;
}

const std::string& type_descriptor::get_index_typestr() const
{
    return index_typestr_;
}


}  // namespace config
}  // namespace gko
