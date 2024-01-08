// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/dim.hpp>


#include <memory>


#include <gtest/gtest.h>


namespace {


TEST(Dim, ConstructsCorrectObject)
{
    gko::dim<2> d{4, 5};

    ASSERT_EQ(d[0], 4);
    ASSERT_EQ(d[1], 5);
}


TEST(Dim, ConstructsCorrectConstexprObject)
{
    constexpr gko::dim<3> d{4, 5, 6};

    ASSERT_EQ(d[0], 4);
    ASSERT_EQ(d[1], 5);
    ASSERT_EQ(d[2], 6);
}


TEST(Dim, ConstructsSquareObject)
{
    gko::dim<2> d{5};

    ASSERT_EQ(d[0], 5);
    ASSERT_EQ(d[1], 5);
}


TEST(Dim, ConstructsNullObject)
{
    gko::dim<2> d{};

    ASSERT_EQ(d[0], 0);
    ASSERT_EQ(d[1], 0);
}


class dim_manager {
public:
    using dim = gko::dim<3>;
    const dim& get_size() const { return size_; }

    static std::unique_ptr<dim_manager> create(const dim& size)
    {
        return std::unique_ptr<dim_manager>{new dim_manager{size}};
    }

private:
    dim_manager(const dim& size) : size_{size} {}
    dim size_;
};


TEST(Dim, CopiesProperlyOnHeap)
{
    auto manager = dim_manager::create(gko::dim<3>{1, 2, 3});

    const auto copy = manager->get_size();

    ASSERT_EQ(copy[0], 1);
    ASSERT_EQ(copy[1], 2);
    ASSERT_EQ(copy[2], 3);
}


TEST(Dim, ConvertsToBool)
{
    gko::dim<2> d1{};
    gko::dim<2> d2{2, 3};

    ASSERT_FALSE(d1);
    ASSERT_TRUE(d2);
}


TEST(Dim, CanAppendToStream1)
{
    gko::dim<2> d2{2, 3};

    std::ostringstream os;
    os << d2;
    ASSERT_EQ(os.str(), "(2, 3)");
}


TEST(Dim, CanAppendToStream2)
{
    gko::dim<3> d2{2, 3, 4};

    std::ostringstream os;
    os << d2;
    ASSERT_EQ(os.str(), "(2, 3, 4)");
}


TEST(Dim, EqualityReturnsTrueWhenEqual)
{
    ASSERT_TRUE(gko::dim<2>(2, 3) == gko::dim<2>(2, 3));
}


TEST(Dim, EqualityReturnsFalseWhenDifferentRows)
{
    ASSERT_FALSE(gko::dim<2>(4, 3) == gko::dim<2>(2, 3));
}


TEST(Dim, EqualityReturnsFalseWhenDifferentColumns)
{
    ASSERT_FALSE(gko::dim<2>(2, 4) == gko::dim<2>(2, 3));
}


TEST(Dim, NotEqualWorks)
{
    ASSERT_TRUE(gko::dim<2>(3, 5) != gko::dim<2>(4, 3));
}


TEST(Dim, MultipliesDimensions)
{
    ASSERT_EQ(gko::dim<2>(2, 3) * gko::dim<2>(4, 5), gko::dim<2>(8, 15));
}


TEST(Dim, TransposesDimensions)
{
    ASSERT_EQ(transpose(gko::dim<2>(3, 5)), gko::dim<2>(5, 3));
}


}  // namespace
