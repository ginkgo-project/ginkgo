// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/config/type_descriptor.hpp>

#include "core/config/type_descriptor_helper.hpp"
#include "core/test/utils.hpp"


using namespace gko::config;


TEST(TypeDescriptor, TemplateCreate)
{
    {
        SCOPED_TRACE("default template");
        auto td = make_type_descriptor<>();

        ASSERT_EQ(td.get_value_typestr(), "float64");
        ASSERT_EQ(td.get_index_typestr(), "int32");
        ASSERT_EQ(td.get_global_index_typestr(), "int64");
    }
    {
        SCOPED_TRACE("specify global indextype");
        auto td = make_type_descriptor<float, int, int>();

        ASSERT_EQ(td.get_value_typestr(), "float32");
        ASSERT_EQ(td.get_index_typestr(), "int32");
        ASSERT_EQ(td.get_global_index_typestr(), "int");
    }
    {
        SCOPED_TRACE("specify valuetype");
        auto td = make_type_descriptor<float>();

        ASSERT_EQ(td.get_value_typestr(), "float32");
        ASSERT_EQ(td.get_index_typestr(), "int32");
        ASSERT_EQ(td.get_global_index_typestr(), "int64");
    }
    {
        SCOPED_TRACE("specify all template");
        auto td =
            make_type_descriptor<std::complex<float>, gko::int64, gko::int64>();

        ASSERT_EQ(td.get_value_typestr(), "complex<float32>");
        ASSERT_EQ(td.get_index_typestr(), "int64");
        ASSERT_EQ(td.get_global_index_typestr(), "int64");
    }
    {
        SCOPED_TRACE("specify void");
        auto td = make_type_descriptor<void, void, void>();

        ASSERT_EQ(td.get_value_typestr(), "void");
        ASSERT_EQ(td.get_index_typestr(), "void");
        ASSERT_EQ(td.get_global_index_typestr(), "void");
    }
}


TEST(TypeDescriptor, Constructor)
{
    {
        SCOPED_TRACE("default constructor");
        type_descriptor td;

        ASSERT_EQ(td.get_value_typestr(), "float64");
        ASSERT_EQ(td.get_index_typestr(), "int32");
    }
    {
        SCOPED_TRACE("specify valuetype");
        type_descriptor td("float32");

        ASSERT_EQ(td.get_value_typestr(), "float32");
        ASSERT_EQ(td.get_index_typestr(), "int32");
    }
    {
        SCOPED_TRACE("specify all parameters");
        type_descriptor td("void", "int64");

        ASSERT_EQ(td.get_value_typestr(), "void");
        ASSERT_EQ(td.get_index_typestr(), "int64");
    }
}
