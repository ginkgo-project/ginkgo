// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/stop/stopping_status.hpp>


#include <gtest/gtest.h>


namespace {


class stopping_status : public ::testing::Test {
protected:
    stopping_status()
    {
        stop_a.reset();
        stop_b.reset();
    }
    gko::stopping_status stop_a;
    gko::stopping_status stop_b;
};


TEST_F(stopping_status, CanStop)
{
    constexpr gko::uint8 id{1};

    stop_a.stop(id, true);
    stop_b.stop(id, false);

    ASSERT_EQ(stop_a.is_finalized(), true);
    ASSERT_EQ(stop_b.is_finalized(), false);
    ASSERT_EQ(stop_a.has_stopped(), true);
    ASSERT_EQ(stop_b.has_stopped(), true);
    ASSERT_EQ(stop_a.get_id(), id);
    ASSERT_EQ(stop_b.get_id(), id);
}


TEST_F(stopping_status, CanConverge)
{
    constexpr gko::uint8 id{5};

    stop_a.converge(id, true);
    stop_b.converge(id, false);

    ASSERT_EQ(stop_a.is_finalized(), true);
    ASSERT_EQ(stop_b.is_finalized(), false);
    ASSERT_EQ(stop_a.has_converged(), true);
    ASSERT_EQ(stop_b.has_converged(), true);
    ASSERT_EQ(stop_a.has_stopped(), true);
    ASSERT_EQ(stop_b.has_stopped(), true);
    ASSERT_EQ(stop_a.get_id(), id);
    ASSERT_EQ(stop_b.get_id(), id);
}


TEST_F(stopping_status, CanCompareEqual)
{
    constexpr gko::uint8 id{5};

    stop_a.converge(id);
    stop_b.converge(id);

    ASSERT_EQ(stop_a == stop_b, true);
    ASSERT_EQ(stop_a != stop_b, false);
}


TEST_F(stopping_status, CanCompareUnequal)
{
    constexpr gko::uint8 id_a{6};
    constexpr gko::uint8 id_b{7};

    stop_a.converge(id_a);
    stop_b.converge(id_b);

    ASSERT_EQ(stop_a == stop_b, false);
    ASSERT_EQ(stop_a != stop_b, true);
}


TEST_F(stopping_status, CanReset)
{
    constexpr gko::uint8 id_a{6};
    constexpr gko::uint8 id_b{7};

    stop_a.converge(id_a, true);
    stop_b.converge(id_b, false);
    stop_a.reset();
    stop_b.reset();

    ASSERT_EQ(stop_a.has_stopped(), false);
    ASSERT_EQ(stop_b.has_stopped(), false);
    ASSERT_EQ(stop_a.has_converged(), false);
    ASSERT_EQ(stop_b.has_converged(), false);
    ASSERT_EQ(stop_a.is_finalized(), false);
    ASSERT_EQ(stop_b.is_finalized(), false);
    ASSERT_EQ(stop_a == stop_b, true);
}


}  // namespace
