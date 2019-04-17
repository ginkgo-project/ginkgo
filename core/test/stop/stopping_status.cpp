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
