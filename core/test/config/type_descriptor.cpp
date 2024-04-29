// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/config/type_descriptor.hpp>


#include <gtest/gtest.h>


#include "core/config/type_descriptor_helper.hpp"
#include "core/test/utils.hpp"


using namespace gko::config;


TEST(TypeDescriptor, TemplateCreate)
{
    {
        SCOPED_TRACE("defaule template");
        auto td = make_type_descriptor<>();

        ASSERT_EQ(td.get_value_typestr(), "double");
        ASSERT_EQ(td.get_index_typestr(), "int");
    }
    {
        SCOPED_TRACE("specify valuetype");
        auto td = make_type_descriptor<float>();

        ASSERT_EQ(td.get_value_typestr(), "float");
        ASSERT_EQ(td.get_index_typestr(), "int");
    }
    {
        SCOPED_TRACE("specify all template");
        auto td = make_type_descriptor<std::complex<float>, gko::int64>();

        ASSERT_EQ(td.get_value_typestr(), "complex<float>");
        ASSERT_EQ(td.get_index_typestr(), "int64");
    }
    {
        SCOPED_TRACE("specify void");
        auto td = make_type_descriptor<void, void>();

        ASSERT_EQ(td.get_value_typestr(), "void");
        ASSERT_EQ(td.get_index_typestr(), "void");
    }
}


TEST(TypeDescriptor, Constructor)
{
    {
        SCOPED_TRACE("defaule constructor");
        type_descriptor td;

        ASSERT_EQ(td.get_value_typestr(), "double");
        ASSERT_EQ(td.get_index_typestr(), "int");
    }
    {
        SCOPED_TRACE("specify valuetype");
        type_descriptor td("float");

        ASSERT_EQ(td.get_value_typestr(), "float");
        ASSERT_EQ(td.get_index_typestr(), "int");
    }
    {
        SCOPED_TRACE("specify all parameters");
        type_descriptor td("void", "int64");

        ASSERT_EQ(td.get_value_typestr(), "void");
        ASSERT_EQ(td.get_index_typestr(), "int64");
    }
}
