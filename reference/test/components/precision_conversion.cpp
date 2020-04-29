/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include "core/components/prefix_sum.hpp"


#include <limits>
#include <memory>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>


#include "core/test/utils.hpp"


namespace {


class PrecisionConversion : public ::testing::Test {
protected:
    PrecisionConversion()
        : ref(gko::ReferenceExecutor::create()),
          rand(293),
          total_size(42793),
          vals(ref, total_size),
          cvals(ref, total_size)
    {
        auto maxval = std::numeric_limits<float>::max();
        std::uniform_real_distribution<float> dist(-maxval, maxval);
        for (gko::size_type i = 0; i < total_size; ++i) {
            vals.get_data()[i] = dist(rand);
            cvals.get_data()[i] = {dist(rand), dist(rand)};
        }
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::default_random_engine rand;
    gko::size_type total_size;
    gko::Array<float> vals;
    gko::Array<std::complex<float>> cvals;
};


TEST_F(PrecisionConversion, ConvertsReal)
{
    gko::Array<double> tmp;
    gko::Array<float> out;

    tmp = vals;
    out = tmp;

    GKO_ASSERT_ARRAY_EQ(&vals, &out);
}


TEST_F(PrecisionConversion, ConvertsRealWithSetExecutor)
{
    gko::Array<double> tmp{ref};
    gko::Array<float> out{ref};

    tmp = vals;
    out = tmp;

    GKO_ASSERT_ARRAY_EQ(&vals, &out);
}


TEST_F(PrecisionConversion, ConvertsRealFromView)
{
    gko::Array<double> tmp{ref};
    gko::Array<float> out{ref};

    tmp = gko::Array<float>::view(ref, vals.get_num_elems(), vals.get_data());
    out = tmp;

    GKO_ASSERT_ARRAY_EQ(&vals, &out);
}


TEST_F(PrecisionConversion, ConvertsComplex)
{
    gko::Array<std::complex<double>> tmp;
    gko::Array<std::complex<float>> out;

    tmp = cvals;
    out = tmp;

    GKO_ASSERT_ARRAY_EQ(&cvals, &out);
}


}  // namespace
