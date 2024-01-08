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


namespace {


class PrecisionConversion : public ::testing::Test {
protected:
    PrecisionConversion()
        : ref(gko::ReferenceExecutor::create()),
          rand(293),
          total_size(42793),
          vals(ref, total_size),
          cvals(ref, total_size),
          vals2(ref, 1),
          expected_float(ref, 1),
          expected_double(ref, 1)
    {
        auto maxval = 1e10f;
        std::uniform_real_distribution<float> dist(-maxval, maxval);
        for (gko::size_type i = 0; i < total_size; ++i) {
            vals.get_data()[i] = dist(rand);
            cvals.get_data()[i] = {dist(rand), dist(rand)};
        }
        gko::uint64 rawdouble{0x4218888000889111ULL};
        gko::uint32 rawfloat{0x50c44400UL};
        gko::uint64 rawrounded{0x4218888000000000ULL};
        std::memcpy(vals2.get_data(), &rawdouble, sizeof(double));
        std::memcpy(expected_float.get_data(), &rawfloat, sizeof(float));
        std::memcpy(expected_double.get_data(), &rawrounded, sizeof(double));
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::default_random_engine rand;
    gko::size_type total_size;
    gko::array<float> vals;
    gko::array<double> vals2;
    gko::array<float> expected_float;
    gko::array<double> expected_double;
    gko::array<std::complex<float>> cvals;
};


TEST_F(PrecisionConversion, ConvertsReal)
{
    gko::array<double> tmp;
    gko::array<float> out;

    tmp = vals;
    out = tmp;

    GKO_ASSERT_ARRAY_EQ(vals, out);
}


TEST_F(PrecisionConversion, ConversionRounds)
{
    gko::array<float> tmp;
    gko::array<double> out;

    tmp = vals2;
    out = tmp;

    GKO_ASSERT_ARRAY_EQ(tmp, expected_float);
    GKO_ASSERT_ARRAY_EQ(out, expected_double);
}


TEST_F(PrecisionConversion, ConvertsRealWithSetExecutor)
{
    gko::array<double> tmp{ref};
    gko::array<float> out{ref};

    tmp = vals;
    out = tmp;

    GKO_ASSERT_ARRAY_EQ(vals, out);
}


TEST_F(PrecisionConversion, ConvertsRealFromView)
{
    gko::array<double> tmp{ref};
    gko::array<float> out{ref};

    tmp = gko::make_array_view(ref, vals.get_size(), vals.get_data());
    out = tmp;

    GKO_ASSERT_ARRAY_EQ(vals, out);
}


TEST_F(PrecisionConversion, ConvertsComplex)
{
    gko::array<std::complex<double>> tmp;
    gko::array<std::complex<float>> out;

    tmp = cvals;
    out = tmp;

    GKO_ASSERT_ARRAY_EQ(cvals, out);
}


}  // namespace
