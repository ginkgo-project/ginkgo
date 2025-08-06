// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/config/type_descriptor.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/types.hpp>

#include "core/config/type_descriptor_helper.hpp"


namespace gko {
namespace config {


type_descriptor update_type(const pnode& config, const type_descriptor& td)
{
    auto value_typestr = td.get_value_typestr();
    auto index_typestr = td.get_index_typestr();
    auto global_index_typestr = td.get_global_index_typestr();

    if (auto& obj = config.get("value_type")) {
        value_typestr = obj.get_string();
    }
    if (auto& obj = config.get("index_type")) {
        GKO_INVALID_STATE(
            "Setting index_type in the config is not allowed. Please set the "
            "proper index_type through type_descriptor of parse");
    }
    if (auto& obj = config.get("global_index_type")) {
        GKO_INVALID_STATE(
            "Setting global_index_type in the config is not allowed. Please "
            "set the proper global_index_type through type_descriptor of "
            "parse");
    }
    return type_descriptor{value_typestr, index_typestr, global_index_typestr};
}


template <typename ValueType, typename IndexType, typename GlobalIndexType>
type_descriptor make_type_descriptor()
{
    return type_descriptor{type_string<ValueType>::str(),
                           type_string<IndexType>::str(),
                           type_string<GlobalIndexType>::str()};
}

#define GKO_DECLARE_MAKE_TYPE_DESCRIPTOR(ValueType, LocalIndexType, \
                                         GlobalIndexType)           \
    type_descriptor                                                 \
    make_type_descriptor<ValueType, LocalIndexType, GlobalIndexType>()
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_MAKE_TYPE_DESCRIPTOR);

#define GKO_DECLARE_MAKE_VOID_TYPE_DESCRIPTOR(LocalIndexType, GlobalIndexType) \
    type_descriptor                                                            \
    make_type_descriptor<void, LocalIndexType, GlobalIndexType>()
GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_MAKE_VOID_TYPE_DESCRIPTOR);


type_descriptor::type_descriptor(std::string value_typestr,
                                 std::string index_typestr,
                                 std::string global_index_typestr)
    : value_typestr_(value_typestr),
      index_typestr_(index_typestr),
      global_index_typestr_(global_index_typestr)
{}

const std::string& type_descriptor::get_value_typestr() const
{
    return value_typestr_;
}

const std::string& type_descriptor::get_index_typestr() const
{
    return index_typestr_;
}

const std::string& type_descriptor::get_local_index_typestr() const
{
    return this->get_index_typestr();
}

const std::string& type_descriptor::get_global_index_typestr() const
{
    return global_index_typestr_;
}

}  // namespace config
}  // namespace gko
