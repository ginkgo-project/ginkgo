// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>


#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


#if !(GINKGO_COMMON_SINGLE_MODE)


class PrecisionConversion : public CommonTestFixture {
protected:
    PrecisionConversion()
        : rand(293),
          total_size(42793),
          vals{ref, total_size},
          cvals{ref, total_size},
          vals2{ref, 1},
          expected_float{ref, 1},
          expected_double{ref, 1},
          dvals{exec},
          dcvals{exec},
          dvals2{exec}
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
