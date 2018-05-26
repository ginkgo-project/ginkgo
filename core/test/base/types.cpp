/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <core/base/types.hpp>


#include <gtest/gtest.h>


namespace {


TEST(Dim, ConstructsCorrectObject)
{
    gko::dim d{4, 5};

    ASSERT_EQ(d.num_rows, 4);
    ASSERT_EQ(d.num_cols, 5);
}


TEST(Dim, ConstructsSquareObject)
{
    gko::dim d{5};

    ASSERT_EQ(d.num_rows, 5);
    ASSERT_EQ(d.num_cols, 5);
}


TEST(Dim, ConstructsNullObject)
{
    gko::dim d{};

    ASSERT_EQ(d.num_rows, 0);
    ASSERT_EQ(d.num_cols, 0);
}


TEST(Dim, ConvertsToBool)
{
    gko::dim d1{};
    gko::dim d2{2, 3};

    ASSERT_FALSE(d1);
    ASSERT_TRUE(d2);
}


TEST(Dim, EqualityReturnsTrueWhenEqual)
{
    ASSERT_TRUE(gko::dim(2, 3) == gko::dim(2, 3));
}


TEST(Dim, EqualityReturnsFalseWhenDifferentRows)
{
    ASSERT_FALSE(gko::dim(4, 3) == gko::dim(2, 3));
}


TEST(Dim, EqualityReturnsFalseWhenDifferentColumns)
{
    ASSERT_FALSE(gko::dim(2, 4) == gko::dim(2, 3));
}


TEST(Dim, NotEqualWorks) { ASSERT_TRUE(gko::dim(3, 5) != gko::dim(4, 3)); }


TEST(Dim, MultipliesDimensions)
{
    ASSERT_EQ(gko::dim(2, 3) * gko::dim(4, 5), gko::dim(8, 15));
}


TEST(Dim, TransposesDimensions)
{
    ASSERT_EQ(transpose(gko::dim(3, 5)), gko::dim(5, 3));
}


}  // namespace
