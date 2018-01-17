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

#include <core/base/mtx_reader.hpp>


#include <gtest/gtest.h>


namespace {


TEST(MtxReader, ReadsDenseRealMtx)
{
    using tpl = std::tuple<gko::int32, gko::int32, double>;

    auto data =
        gko::read_raw_from_mtx<double, gko::int32>("data/dense_real.mtx");

    ASSERT_EQ(data.num_rows, 2);
    ASSERT_EQ(data.num_cols, 3);
    auto &v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, 1.0));
    ASSERT_EQ(v[1], tpl(0, 1, 3.0));
    ASSERT_EQ(v[2], tpl(0, 2, 2.0));
    ASSERT_EQ(v[3], tpl(1, 0, 0.0));
    ASSERT_EQ(v[4], tpl(1, 1, 5.0));
    ASSERT_EQ(v[5], tpl(1, 2, 0.0));
}


TEST(MtxReader, ReadsDenseIntegerMtx)
{
    using tpl = std::tuple<gko::int32, gko::int32, double>;

    auto data =
        gko::read_raw_from_mtx<double, gko::int32>("data/dense_integer.mtx");

    ASSERT_EQ(data.num_rows, 2);
    ASSERT_EQ(data.num_cols, 3);
    auto &v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, 1.0));
    ASSERT_EQ(v[1], tpl(0, 1, 3.0));
    ASSERT_EQ(v[2], tpl(0, 2, 2.0));
    ASSERT_EQ(v[3], tpl(1, 0, 0.0));
    ASSERT_EQ(v[4], tpl(1, 1, 5.0));
    ASSERT_EQ(v[5], tpl(1, 2, 0.0));
}


TEST(MtxReader, ReadsDenseComplexMtx)
{
    using cpx = std::complex<double>;
    using tpl = std::tuple<gko::int32, gko::int32, cpx>;

    auto data =
        gko::read_raw_from_mtx<cpx, gko::int32>("data/dense_complex.mtx");

    ASSERT_EQ(data.num_rows, 2);
    ASSERT_EQ(data.num_cols, 3);
    auto &v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, cpx(1.0, 2.0)));
    ASSERT_EQ(v[1], tpl(0, 1, cpx(3.0, 1.0)));
    ASSERT_EQ(v[2], tpl(0, 2, cpx(2.0, 4.0)));
    ASSERT_EQ(v[3], tpl(1, 0, cpx(0.0, 0.0)));
    ASSERT_EQ(v[4], tpl(1, 1, cpx(5.0, 3.0)));
    ASSERT_EQ(v[5], tpl(1, 2, cpx(0.0, 0.0)));
}


TEST(MtxReader, ReadsSparseRealMtx)
{
    using tpl = std::tuple<gko::int32, gko::int32, double>;

    auto data =
        gko::read_raw_from_mtx<double, gko::int32>("data/sparse_real.mtx");

    ASSERT_EQ(data.num_rows, 2);
    ASSERT_EQ(data.num_cols, 3);
    auto &v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, 1.0));
    ASSERT_EQ(v[1], tpl(0, 1, 3.0));
    ASSERT_EQ(v[2], tpl(0, 2, 2.0));
    ASSERT_EQ(v[3], tpl(1, 1, 5.0));
}


TEST(MtxReader, ReadsSparseRealSymetricMtx)
{
    using tpl = std::tuple<gko::int32, gko::int32, double>;

    auto data = gko::read_raw_from_mtx<double, gko::int32>(
        "data/sparse_real_symmetric.mtx");

    ASSERT_EQ(data.num_rows, 3);
    ASSERT_EQ(data.num_cols, 3);
    auto &v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, 1.0));
    ASSERT_EQ(v[1], tpl(0, 1, 2.0));
    ASSERT_EQ(v[2], tpl(0, 2, 3.0));
    ASSERT_EQ(v[3], tpl(1, 0, 2.0));
    ASSERT_EQ(v[4], tpl(2, 0, 3.0));
    ASSERT_EQ(v[5], tpl(2, 2, 6.0));
}


TEST(MtxReader, ReadsSparseRealSkewSymetricMtx)
{
    using tpl = std::tuple<gko::int32, gko::int32, double>;

    auto data = gko::read_raw_from_mtx<double, gko::int32>(
        "data/sparse_real_skew_symmetric.mtx");

    ASSERT_EQ(data.num_rows, 3);
    ASSERT_EQ(data.num_cols, 3);
    auto &v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 1, -2.0));
    ASSERT_EQ(v[1], tpl(0, 2, -3.0));
    ASSERT_EQ(v[2], tpl(1, 0, 2.0));
    ASSERT_EQ(v[3], tpl(2, 0, 3.0));
}


TEST(MtxReader, ReadsSparsePatternMtx)
{
    using tpl = std::tuple<gko::int32, gko::int32, double>;

    auto data =
        gko::read_raw_from_mtx<double, gko::int32>("data/sparse_pattern.mtx");

    ASSERT_EQ(data.num_rows, 2);
    ASSERT_EQ(data.num_cols, 3);
    auto &v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, 0.0));
    ASSERT_EQ(v[1], tpl(0, 1, 0.0));
    ASSERT_EQ(v[2], tpl(0, 2, 0.0));
    ASSERT_EQ(v[3], tpl(1, 1, 0.0));
}


TEST(MtxReader, ReadsSparseComplexMtx)
{
    using cpx = std::complex<double>;
    using tpl = std::tuple<gko::int32, gko::int32, cpx>;

    auto data =
        gko::read_raw_from_mtx<cpx, gko::int32>("data/sparse_complex.mtx");

    ASSERT_EQ(data.num_rows, 2);
    ASSERT_EQ(data.num_cols, 3);
    auto &v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, cpx(1.0, 2.0)));
    ASSERT_EQ(v[1], tpl(0, 1, cpx(3.0, 1.0)));
    ASSERT_EQ(v[2], tpl(0, 2, cpx(2.0, 4.0)));
    ASSERT_EQ(v[3], tpl(1, 1, cpx(5.0, 3.0)));
}


TEST(MtxReader, ReadsSparseComplexHermitianMtx)
{
    using cpx = std::complex<double>;
    using tpl = std::tuple<gko::int32, gko::int32, cpx>;

    auto data = gko::read_raw_from_mtx<cpx, gko::int32>(
        "data/sparse_complex_hermitian.mtx");

    ASSERT_EQ(data.num_rows, 2);
    ASSERT_EQ(data.num_cols, 3);
    auto &v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 1, cpx(3.0, 1.0)));
    ASSERT_EQ(v[1], tpl(0, 2, cpx(2.0, 4.0)));
    ASSERT_EQ(v[2], tpl(1, 0, cpx(3.0, -1.0)));
    ASSERT_EQ(v[3], tpl(2, 0, cpx(2.0, -4.0)));
}


}  // namespace
