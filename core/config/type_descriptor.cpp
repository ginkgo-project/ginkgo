// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/config/type_descriptor.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>

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

// global_index: void
template type_descriptor make_type_descriptor<void, void, void>();
template type_descriptor make_type_descriptor<float, void, void>();
template type_descriptor make_type_descriptor<double, void, void>();
template type_descriptor
make_type_descriptor<std::complex<float>, void, void>();
template type_descriptor
make_type_descriptor<std::complex<double>, void, void>();
template type_descriptor make_type_descriptor<void, int32, void>();
template type_descriptor make_type_descriptor<float, int32, void>();
template type_descriptor make_type_descriptor<double, int32, void>();
template type_descriptor
make_type_descriptor<std::complex<float>, int32, void>();
template type_descriptor
make_type_descriptor<std::complex<double>, int32, void>();
template type_descriptor make_type_descriptor<void, int64, void>();
template type_descriptor make_type_descriptor<float, int64, void>();
template type_descriptor make_type_descriptor<double, int64, void>();
template type_descriptor
make_type_descriptor<std::complex<float>, int64, void>();
template type_descriptor
make_type_descriptor<std::complex<double>, int64, void>();

// global_index int32
template type_descriptor make_type_descriptor<void, void, int32>();
template type_descriptor make_type_descriptor<float, void, int32>();
template type_descriptor make_type_descriptor<double, void, int32>();
template type_descriptor
make_type_descriptor<std::complex<float>, void, int32>();
template type_descriptor
make_type_descriptor<std::complex<double>, void, int32>();
template type_descriptor make_type_descriptor<void, int32, int32>();
template type_descriptor make_type_descriptor<float, int32, int32>();
template type_descriptor make_type_descriptor<double, int32, int32>();
template type_descriptor
make_type_descriptor<std::complex<float>, int32, int32>();
template type_descriptor
make_type_descriptor<std::complex<double>, int32, int32>();

// global_index_type int64
template type_descriptor make_type_descriptor<void, void, int64>();
template type_descriptor make_type_descriptor<float, void, int64>();
template type_descriptor make_type_descriptor<double, void, int64>();
template type_descriptor
make_type_descriptor<std::complex<float>, void, int64>();
template type_descriptor
make_type_descriptor<std::complex<double>, void, int64>();
template type_descriptor make_type_descriptor<void, int32, int64>();
template type_descriptor make_type_descriptor<float, int32, int64>();
template type_descriptor make_type_descriptor<double, int32, int64>();
template type_descriptor
make_type_descriptor<std::complex<float>, int32, int64>();
template type_descriptor
make_type_descriptor<std::complex<double>, int32, int64>();
template type_descriptor make_type_descriptor<void, int64, int64>();
template type_descriptor make_type_descriptor<float, int64, int64>();
template type_descriptor make_type_descriptor<double, int64, int64>();
template type_descriptor
make_type_descriptor<std::complex<float>, int64, int64>();
template type_descriptor
make_type_descriptor<std::complex<double>, int64, int64>();


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

const std::string& type_descriptor::get_global_index_typestr() const
{
    return global_index_typestr_;
}

}  // namespace config
}  // namespace gko
