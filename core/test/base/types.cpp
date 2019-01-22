/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2019

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

#include <ginkgo/core/base/types.hpp>


#include <gtest/gtest.h>


namespace {


TEST(PrecisionReduction, CreatesDefaultEncoding)
{
    auto e = gko::precision_reduction();

    ASSERT_EQ(e.get_preserving(), 0);
    ASSERT_EQ(e.get_nonpreserving(), 0);
}


TEST(PrecisionReduction, CreatesCustomEncoding)
{
    auto e = gko::precision_reduction(2, 4);

    ASSERT_EQ(e.get_preserving(), 2);
    ASSERT_EQ(e.get_nonpreserving(), 4);
}


TEST(PrecisionReduction, ComparesEncodings)
{
    auto x = gko::precision_reduction(1, 2);
    auto y = gko::precision_reduction(1, 2);
    auto z = gko::precision_reduction(3, 1);

    ASSERT_TRUE(x == y);
    ASSERT_TRUE(!(x == z));
    ASSERT_TRUE(!(y == z));
    ASSERT_TRUE(!(x != y));
    ASSERT_TRUE(x != z);
    ASSERT_TRUE(y != z);
}


TEST(PrecisionReduction, CreatesAutodetectEncoding)
{
    auto ad = gko::precision_reduction::autodetect();

    ASSERT_NE(ad, gko::precision_reduction());
    ASSERT_NE(ad, gko::precision_reduction(1, 2));
}


TEST(PrecisionReduction, ConvertsToStorageType)
{
    auto st = static_cast<gko::precision_reduction::storage_type>(
        gko::precision_reduction{});

    ASSERT_EQ(st, 0);
}


TEST(PrecisionReduction, ComputesCommonEncoding)
{
    auto e1 = gko::precision_reduction(2, 3);
    auto e2 = gko::precision_reduction(3, 1);

    ASSERT_EQ(gko::precision_reduction::common(e1, e2),
              gko::precision_reduction(2, 1));
}


}  // namespace
