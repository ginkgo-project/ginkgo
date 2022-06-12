/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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
#include "test/utils/executor.hpp"

namespace {


#if !(GINKGO_COMMON_SINGLE_MODE)


class PrecisionConversion : public ::testing::Test {
protected:
    PrecisionConversion() : rand(293), total_size(42793) {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        init_executor(ref, exec);
        vals = gko::array<float>{ref, total_size};
        cvals = gko::array<std::complex<float>>{ref, total_size};
        vals2 = gko::array<double>{ref, 1};
        expected_float = gko::array<float>{ref, 1};
        expected_double = gko::array<double>{ref, 1};
        dvals = gko::array<float>{exec};
        dcvals = gko::array<std::complex<float>>{exec};
        dvals2 = gko::array<double>{exec};
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

    void TearDown()
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;
    std::default_random_engine rand;
    gko::size_type total_size;
    gko::array<float> vals;
    gko::array<float> dvals;
    gko::array<double> vals2;
    gko::array<double> dvals2;
    gko::array<float> expected_float;
    gko::array<double> expected_double;
    gko::array<std::complex<float>> cvals;
    gko::array<std::complex<float>> dcvals;
};


TEST_F(PrecisionConversion, ConvertsReal)
{
    gko::array<double> dtmp;
    gko::array<float> dout;

    dtmp = dvals;
    dout = dtmp;

    GKO_ASSERT_ARRAY_EQ(dvals, dout);
}


TEST_F(PrecisionConversion, ConvertsRealViaRef)
{
    gko::array<double> tmp{ref};
    gko::array<float> dout;

    tmp = dvals;
    dout = tmp;

    GKO_ASSERT_ARRAY_EQ(dvals, dout);
}


TEST_F(PrecisionConversion, ConvertsComplex)
{
    gko::array<std::complex<double>> dtmp;
    gko::array<std::complex<float>> dout;

    dtmp = dcvals;
    dout = dtmp;

    GKO_ASSERT_ARRAY_EQ(dcvals, dout);
}


TEST_F(PrecisionConversion, ConversionRounds)
{
    gko::array<float> dtmp;
    gko::array<double> dout;

    dtmp = dvals2;
    dout = dtmp;

    GKO_ASSERT_ARRAY_EQ(dtmp, expected_float);
    GKO_ASSERT_ARRAY_EQ(dout, expected_double);
}


TEST_F(PrecisionConversion, ConvertsRealFromRef)
{
    gko::array<double> dtmp;
    gko::array<float> dout;

    dtmp = vals;
    dout = dtmp;

    GKO_ASSERT_ARRAY_EQ(dvals, dout);
}


TEST_F(PrecisionConversion, ConvertsComplexFromRef)
{
    gko::array<std::complex<double>> dtmp;
    gko::array<std::complex<float>> dout;

    dtmp = cvals;
    dout = dtmp;

    GKO_ASSERT_ARRAY_EQ(dcvals, dout);
}


#endif


}  // namespace
