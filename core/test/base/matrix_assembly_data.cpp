// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/matrix_assembly_data.hpp>


#include <gtest/gtest.h>


namespace {


TEST(MatrixAssemblyData, InitializesWithZeros)
{
    gko::matrix_assembly_data<double, int> m(gko::dim<2>{3, 5});

    ASSERT_EQ(m.get_size(), gko::dim<2>(3, 5));
    ASSERT_EQ(m.get_num_stored_elements(), 0);
    ASSERT_EQ(m.get_value(0, 0), 0.0);
    ASSERT_FALSE(m.contains(0, 0));
}


TEST(MatrixAssemblyData, InsertsValuesWithoutAdding)
{
    gko::matrix_assembly_data<double, int> m(gko::dim<2>{3, 5});

    m.add_value(0, 0, 1.3);
    m.add_value(2, 3, 2.2);
    m.add_value(1, 4, 1.1);
    m.add_value(1, 2, 3.6);

    ASSERT_EQ(m.get_size(), gko::dim<2>(3, 5));
    ASSERT_EQ(m.get_num_stored_elements(), 4);
    ASSERT_EQ(m.get_value(0, 0), 1.3);
    ASSERT_EQ(m.get_value(2, 3), 2.2);
    ASSERT_EQ(m.get_value(1, 4), 1.1);
    ASSERT_EQ(m.get_value(1, 2), 3.6);
    ASSERT_TRUE(m.contains(0, 0));
}


TEST(MatrixAssemblyData, InsertsValuesWithAdding)
{
    gko::matrix_assembly_data<double, int> m(gko::dim<2>{3, 5});

    m.add_value(0, 0, 1.3);
    m.add_value(2, 3, 2.2);
    m.add_value(1, 4, 1.1);
    m.add_value(1, 2, 3.6);
    m.add_value(1, 4, 9.1);
    m.add_value(2, 3, 1.3);

    ASSERT_EQ(m.get_size(), gko::dim<2>(3, 5));
    ASSERT_EQ(m.get_num_stored_elements(), 4);
    ASSERT_EQ(m.get_value(0, 0), 1.3);
    ASSERT_EQ(m.get_value(2, 3), 3.5);
    ASSERT_EQ(m.get_value(1, 4), 10.2);
    ASSERT_EQ(m.get_value(1, 2), 3.6);
}


TEST(MatrixAssemblyData, OverwritesValuesWhenNotAdding)
{
    gko::matrix_assembly_data<double, int> m(gko::dim<2>{3, 5});

    m.set_value(0, 0, 1.3);
    m.set_value(2, 3, 2.2);
    m.set_value(1, 4, 1.1);
    m.set_value(1, 2, 3.6);
    m.set_value(1, 4, 9.1);
    m.set_value(2, 3, 1.4);

    ASSERT_EQ(m.get_size(), gko::dim<2>(3, 5));
    ASSERT_EQ(m.get_num_stored_elements(), 4);
    ASSERT_EQ(m.get_value(0, 0), 1.3);
    ASSERT_EQ(m.get_value(2, 3), 1.4);
    ASSERT_EQ(m.get_value(1, 4), 9.1);
    ASSERT_EQ(m.get_value(1, 2), 3.6);
}


TEST(MatrixAssemblyData, GetsSortedData)
{
    gko::matrix_assembly_data<double, int> m(gko::dim<2>{3, 5});
    std::vector<gko::matrix_data<double, int>::nonzero_type> reference{
        {0, 0, 1.3}, {1, 2, 3.6}, {1, 4, 1.1}, {2, 3, 2.2}};
    m.set_value(0, 0, 1.3);
    m.set_value(2, 3, 2.2);
    m.set_value(1, 4, 1.1);
    m.set_value(1, 2, 3.6);

    auto sorted = m.get_ordered_data();

    ASSERT_EQ(sorted.size, m.get_size());
    ASSERT_EQ(sorted.nonzeros, reference);
}


}  // namespace
