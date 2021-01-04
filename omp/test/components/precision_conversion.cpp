/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <cstring>
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
          exec(gko::OmpExecutor::create()),
          rand(293),
          total_size(42793),
          vals(ref, total_size),
          cvals(ref, total_size),
          vals2(ref, 1),
          expected_float(ref, 1),
          expected_double(ref, 1),
          dvals(exec),
          dcvals(exec),
          dvals2(exec)
    {
        auto maxval = 1e10f;
        std::uniform_real_distribution<float> dist(-maxval, maxval);
        for (gko::size_type i = 0; i < total_size; ++i) {
            vals.get_data()[i] = dist(rand);
            cvals.get_data()[i] = {dist(rand), dist(rand)};
        }
        dvals = vals;
        dcvals = cvals;
        gko::uint64 rawdouble{0x4218888000889111ULL};
        gko::uint32 rawfloat{0x50c44400UL};
        gko::uint64 rawrounded{0x4218888000000000ULL};
        std::memcpy(vals2.get_data(), &rawdouble, sizeof(double));
        std::memcpy(expected_float.get_data(), &rawfloat, sizeof(float));
        std::memcpy(expected_double.get_data(), &rawrounded, sizeof(double));
        dvals2 = vals2;
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::OmpExecutor> exec;
    std::default_random_engine rand;
    gko::size_type total_size;
    gko::Array<float> vals;
    gko::Array<float> dvals;
    gko::Array<double> vals2;
    gko::Array<double> dvals2;
    gko::Array<float> expected_float;
    gko::Array<double> expected_double;
    gko::Array<std::complex<float>> cvals;
    gko::Array<std::complex<float>> dcvals;
};


TEST_F(PrecisionConversion, ConvertsReal)
{
    gko::Array<double> dtmp;
    gko::Array<float> dout;

    dtmp = dvals;
    dout = dtmp;

    GKO_ASSERT_ARRAY_EQ(dvals, dout);
}


TEST_F(PrecisionConversion, ConvertsRealViaRef)
{
    gko::Array<double> tmp{ref};
    gko::Array<float> dout;

    tmp = dvals;
    dout = tmp;

    GKO_ASSERT_ARRAY_EQ(dvals, dout);
}


TEST_F(PrecisionConversion, ConvertsComplex)
{
    gko::Array<std::complex<double>> dtmp;
    gko::Array<std::complex<float>> dout;

    dtmp = dcvals;
    dout = dtmp;

    GKO_ASSERT_ARRAY_EQ(dcvals, dout);
}


TEST_F(PrecisionConversion, ConversionRounds)
{
    gko::Array<float> dtmp;
    gko::Array<double> dout;

    dtmp = dvals2;
    dout = dtmp;

    GKO_ASSERT_ARRAY_EQ(dtmp, expected_float);
    GKO_ASSERT_ARRAY_EQ(dout, expected_double);
}


TEST_F(PrecisionConversion, ConvertsRealFromRef)
{
    gko::Array<double> dtmp;
    gko::Array<float> dout;

    dtmp = vals;
    dout = dtmp;

    GKO_ASSERT_ARRAY_EQ(dvals, dout);
}


TEST_F(PrecisionConversion, ConvertsComplexFromRef)
{
    gko::Array<std::complex<double>> dtmp;
    gko::Array<std::complex<float>> dout;

    dtmp = cvals;
    dout = dtmp;

    GKO_ASSERT_ARRAY_EQ(dcvals, dout);
}


}  // namespace
