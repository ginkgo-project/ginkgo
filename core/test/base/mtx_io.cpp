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

#include <ginkgo/core/base/mtx_io.hpp>


#include <sstream>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/lin_op.hpp>


#include "core/test/utils.hpp"


namespace {


TEST(MtxReader, ReadsDenseDoubleRealMtx)
{
    using tpl = gko::matrix_data<double, gko::int32>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array real general\n"
        "2 3\n"
        "1.0\n"
        "0.0\n"
        "3.0\n"
        "5.0\n"
        "2.0\n"
        "0.0\n");

    auto data = gko::read_raw<double, gko::int32>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    auto &v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, 1.0));
    ASSERT_EQ(v[1], tpl(0, 1, 3.0));
    ASSERT_EQ(v[2], tpl(0, 2, 2.0));
    ASSERT_EQ(v[3], tpl(1, 0, 0.0));
    ASSERT_EQ(v[4], tpl(1, 1, 5.0));
    ASSERT_EQ(v[5], tpl(1, 2, 0.0));
}


TEST(MtxReader, ReadsDenseDoubleRealMtxWith64Index)
{
    using tpl = gko::matrix_data<double, gko::int64>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array real general\n"
        "2 3\n"
        "1.0\n"
        "0.0\n"
        "3.0\n"
        "5.0\n"
        "2.0\n"
        "0.0\n");

    auto data = gko::read_raw<double, gko::int64>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    auto &v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, 1.0));
    ASSERT_EQ(v[1], tpl(0, 1, 3.0));
    ASSERT_EQ(v[2], tpl(0, 2, 2.0));
    ASSERT_EQ(v[3], tpl(1, 0, 0.0));
    ASSERT_EQ(v[4], tpl(1, 1, 5.0));
    ASSERT_EQ(v[5], tpl(1, 2, 0.0));
}


TEST(MtxReader, ReadsDenseFloatIntegerMtx)
{
    using tpl = gko::matrix_data<float, gko::int32>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array integer general\n"
        "2 3\n"
        "1\n"
        "0\n"
        "3\n"
        "5\n"
        "2\n"
        "0\n");

    auto data = gko::read_raw<float, gko::int32>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    auto &v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, 1.0));
    ASSERT_EQ(v[1], tpl(0, 1, 3.0));
    ASSERT_EQ(v[2], tpl(0, 2, 2.0));
    ASSERT_EQ(v[3], tpl(1, 0, 0.0));
    ASSERT_EQ(v[4], tpl(1, 1, 5.0));
    ASSERT_EQ(v[5], tpl(1, 2, 0.0));
}


TEST(MtxReader, ReadsDenseFloatIntegerMtxWith64Index)
{
    using tpl = gko::matrix_data<float, gko::int64>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array integer general\n"
        "2 3\n"
        "1\n"
        "0\n"
        "3\n"
        "5\n"
        "2\n"
        "0\n");

    auto data = gko::read_raw<float, gko::int64>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    auto &v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, 1.0));
    ASSERT_EQ(v[1], tpl(0, 1, 3.0));
    ASSERT_EQ(v[2], tpl(0, 2, 2.0));
    ASSERT_EQ(v[3], tpl(1, 0, 0.0));
    ASSERT_EQ(v[4], tpl(1, 1, 5.0));
    ASSERT_EQ(v[5], tpl(1, 2, 0.0));
}


TEST(MtxReader, ReadsDenseComplexDoubleMtx)
{
    using cpx = std::complex<double>;
    using tpl = gko::matrix_data<cpx, gko::int32>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array complex general\n"
        "2 3\n"
        "1.0 2.0\n"
        "0.0 0.0\n"
        "3.0 1.0\n"
        "5.0 3.0\n"
        "2.0 4.0\n"
        "0.0 0.0\n");

    auto data = gko::read_raw<cpx, gko::int32>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    auto &v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, cpx(1.0, 2.0)));
    ASSERT_EQ(v[1], tpl(0, 1, cpx(3.0, 1.0)));
    ASSERT_EQ(v[2], tpl(0, 2, cpx(2.0, 4.0)));
    ASSERT_EQ(v[3], tpl(1, 0, cpx(0.0, 0.0)));
    ASSERT_EQ(v[4], tpl(1, 1, cpx(5.0, 3.0)));
    ASSERT_EQ(v[5], tpl(1, 2, cpx(0.0, 0.0)));
}


TEST(MtxReader, ReadsDenseComplexDoubleMtxWith64Index)
{
    using cpx = std::complex<double>;
    using tpl = gko::matrix_data<cpx, gko::int64>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array complex general\n"
        "2 3\n"
        "1.0 2.0\n"
        "0.0 0.0\n"
        "3.0 1.0\n"
        "5.0 3.0\n"
        "2.0 4.0\n"
        "0.0 0.0\n");

    auto data = gko::read_raw<cpx, gko::int64>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    auto &v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, cpx(1.0, 2.0)));
    ASSERT_EQ(v[1], tpl(0, 1, cpx(3.0, 1.0)));
    ASSERT_EQ(v[2], tpl(0, 2, cpx(2.0, 4.0)));
    ASSERT_EQ(v[3], tpl(1, 0, cpx(0.0, 0.0)));
    ASSERT_EQ(v[4], tpl(1, 1, cpx(5.0, 3.0)));
    ASSERT_EQ(v[5], tpl(1, 2, cpx(0.0, 0.0)));
}


TEST(MtxReader, ReadsDenseComplexFloatMtx)
{
    using cpx = std::complex<float>;
    using tpl = gko::matrix_data<cpx, gko::int32>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array complex general\n"
        "2 3\n"
        "1.0 2.0\n"
        "0.0 0.0\n"
        "3.0 1.0\n"
        "5.0 3.0\n"
        "2.0 4.0\n"
        "0.0 0.0\n");

    auto data = gko::read_raw<cpx, gko::int32>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    auto &v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, cpx(1.0, 2.0)));
    ASSERT_EQ(v[1], tpl(0, 1, cpx(3.0, 1.0)));
    ASSERT_EQ(v[2], tpl(0, 2, cpx(2.0, 4.0)));
    ASSERT_EQ(v[3], tpl(1, 0, cpx(0.0, 0.0)));
    ASSERT_EQ(v[4], tpl(1, 1, cpx(5.0, 3.0)));
    ASSERT_EQ(v[5], tpl(1, 2, cpx(0.0, 0.0)));
}


TEST(MtxReader, ReadsDenseComplexFloatMtxWith64Index)
{
    using cpx = std::complex<float>;
    using tpl = gko::matrix_data<cpx, gko::int64>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array complex general\n"
        "2 3\n"
        "1.0 2.0\n"
        "0.0 0.0\n"
        "3.0 1.0\n"
        "5.0 3.0\n"
        "2.0 4.0\n"
        "0.0 0.0\n");

    auto data = gko::read_raw<cpx, gko::int64>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
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
    using tpl = gko::matrix_data<double, gko::int32>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix coordinate real general\n"
        "2 3 4\n"
        "1 1 1.0\n"
        "2 2 5.0\n"
        "1 2 3.0\n"
        "1 3 2.0\n");

    auto data = gko::read_raw<double, gko::int32>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    auto &v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, 1.0));
    ASSERT_EQ(v[1], tpl(0, 1, 3.0));
    ASSERT_EQ(v[2], tpl(0, 2, 2.0));
    ASSERT_EQ(v[3], tpl(1, 1, 5.0));
}


TEST(MtxReader, ReadsSparseRealSymetricMtx)
{
    using tpl = gko::matrix_data<double, gko::int32>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix coordinate real symmetric\n"
        "3 3 4\n"
        "1 1 1.0\n"
        "2 1 2.0\n"
        "3 3 6.0\n"
        "3 1 3.0\n");

    auto data = gko::read_raw<double, gko::int32>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(3, 3));
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
    using tpl = gko::matrix_data<double, gko::int32>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix coordinate real skew-symmetric\n"
        "3 3 2\n"
        "2 1 2.0\n"
        "3 1 3.0\n");

    auto data = gko::read_raw<double, gko::int32>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(3, 3));
    auto &v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 1, -2.0));
    ASSERT_EQ(v[1], tpl(0, 2, -3.0));
    ASSERT_EQ(v[2], tpl(1, 0, 2.0));
    ASSERT_EQ(v[3], tpl(2, 0, 3.0));
}


TEST(MtxReader, ReadsSparsePatternMtx)
{
    using tpl = gko::matrix_data<double, gko::int32>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix coordinate pattern general\n"
        "2 3 4\n"
        "1 1\n"
        "2 2\n"
        "1 2\n"
        "1 3\n");

    auto data = gko::read_raw<double, gko::int32>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    auto &v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, 1.0));
    ASSERT_EQ(v[1], tpl(0, 1, 1.0));
    ASSERT_EQ(v[2], tpl(0, 2, 1.0));
    ASSERT_EQ(v[3], tpl(1, 1, 1.0));
}


TEST(MtxReader, ReadsSparseComplexMtx)
{
    using cpx = std::complex<double>;
    using tpl = gko::matrix_data<cpx, gko::int32>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix coordinate complex general\n"
        "2 3 4\n"
        "1 1 1.0 2.0\n"
        "2 2 5.0 3.0\n"
        "1 2 3.0 1.0\n"
        "1 3 2.0 4.0\n");

    auto data = gko::read_raw<cpx, gko::int32>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    auto &v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, cpx(1.0, 2.0)));
    ASSERT_EQ(v[1], tpl(0, 1, cpx(3.0, 1.0)));
    ASSERT_EQ(v[2], tpl(0, 2, cpx(2.0, 4.0)));
    ASSERT_EQ(v[3], tpl(1, 1, cpx(5.0, 3.0)));
}


TEST(MtxReader, ReadsSparseComplexHermitianMtx)
{
    using cpx = std::complex<double>;
    using tpl = gko::matrix_data<cpx, gko::int32>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix coordinate complex hermitian\n"
        "2 3 2\n"
        "1 2 3.0 1.0\n"
        "1 3 2.0 4.0\n");

    auto data = gko::read_raw<cpx, gko::int32>(iss);

    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    auto &v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 1, cpx(3.0, 1.0)));
    ASSERT_EQ(v[1], tpl(0, 2, cpx(2.0, 4.0)));
    ASSERT_EQ(v[2], tpl(1, 0, cpx(3.0, -1.0)));
    ASSERT_EQ(v[3], tpl(2, 0, cpx(2.0, -4.0)));
}


TEST(MtxReader, FailsWhenReadingSparseComplexMtxToRealMtx)
{
    using cpx = std::complex<double>;
    using tpl = gko::matrix_data<cpx, gko::int32>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix coordinate complex general\n"
        "2 3 4\n"
        "1 1 1.0 2.0\n"
        "2 2 5.0 3.0\n"
        "1 2 3.0 1.0\n"
        "1 3 2.0 4.0\n");

    ASSERT_THROW((gko::read_raw<double, gko::int32>(iss)), gko::StreamError);
}


TEST(MatrixData, WritesDoubleRealMatrixToMatrixMarketArray)
{
    // clang-format off
    gko::matrix_data<double, gko::int32> data{
        {1.0, 2.0},
        {2.1, 0.0},
        {3.0, 3.2}};
    // clang-format on
    std::ostringstream oss{};

    write_raw(oss, data);

    ASSERT_EQ(oss.str(),
              "%%MatrixMarket matrix array real general\n"
              "3 2\n"
              "1\n"
              "2.1\n"
              "3\n"
              "2\n"
              "0\n"
              "3.2\n");
}


TEST(MatrixData, WritesFloatRealMatrixToMatrixMarketCoordinate)
{
    // clang-format off
    gko::matrix_data<float, gko::int32> data{
        {1.0, 2.0},
        {2.1, 0.0},
        {3.0, 3.2}};
    // clang-format on
    std::ostringstream oss{};

    write_raw(oss, data, gko::layout_type::coordinate);

    ASSERT_EQ(oss.str(),
              "%%MatrixMarket matrix coordinate real general\n"
              "3 2 5\n"
              "1 1 1\n"
              "1 2 2\n"
              "2 1 2.1\n"
              "3 1 3\n"
              "3 2 3.2\n");
}


TEST(MatrixData, WritesDoubleRealMatrixToMatrixMarketArrayWith64Index)
{
    // clang-format off
    gko::matrix_data<double, gko::int64> data{
        {1.0, 2.0},
        {2.1, 0.0},
        {3.0, 3.2}};
    // clang-format on
    std::ostringstream oss{};

    write_raw(oss, data);

    ASSERT_EQ(oss.str(),
              "%%MatrixMarket matrix array real general\n"
              "3 2\n"
              "1\n"
              "2.1\n"
              "3\n"
              "2\n"
              "0\n"
              "3.2\n");
}


TEST(MatrixData, WritesFloatRealMatrixToMatrixMarketCoordinateWith64Index)
{
    // clang-format off
    gko::matrix_data<float, gko::int64> data{
        {1.0, 2.0},
        {2.1, 0.0},
        {3.0, 3.2}};
    // clang-format on
    std::ostringstream oss{};

    write_raw(oss, data, gko::layout_type::coordinate);

    ASSERT_EQ(oss.str(),
              "%%MatrixMarket matrix coordinate real general\n"
              "3 2 5\n"
              "1 1 1\n"
              "1 2 2\n"
              "2 1 2.1\n"
              "3 1 3\n"
              "3 2 3.2\n");
}


TEST(MatrixData, WritesComplexDoubleMatrixToMatrixMarketArray)
{
    // clang-format off
    gko::matrix_data<std::complex<double>, gko::int32> data{
        {{1.0, 0.0}, {2.0, 3.2}},
        {{2.1, 2.2}, {0.0, 0.0}},
        {{0.0, 3.0}, {3.2, 5.3}}};
    // clang-format on
    std::ostringstream oss{};

    write_raw(oss, data);

    ASSERT_EQ(oss.str(),
              "%%MatrixMarket matrix array complex general\n"
              "3 2\n"
              "1 0\n"
              "2.1 2.2\n"
              "0 3\n"
              "2 3.2\n"
              "0 0\n"
              "3.2 5.3\n");
}


TEST(MatrixData, WritesComplexFloatMatrixToMatrixMarketCoordinate)
{
    // clang-format off
    gko::matrix_data<std::complex<float>, gko::int32> data{
        {{1.0, 0.0}, {2.0, 3.2}},
        {{2.1, 2.2}, {0.0, 0.0}},
        {{0.0, 3.0}, {3.2, 5.3}}};
    // clang-format on
    std::ostringstream oss{};

    write_raw(oss, data, gko::layout_type::coordinate);

    ASSERT_EQ(oss.str(),
              "%%MatrixMarket matrix coordinate complex general\n"
              "3 2 5\n"
              "1 1 1 0\n"
              "1 2 2 3.2\n"
              "2 1 2.1 2.2\n"
              "3 1 0 3\n"
              "3 2 3.2 5.3\n");
}


TEST(MatrixData, WritesComplexDoubleMatrixToMatrixMarketArrayWith64Index)
{
    // clang-format off
    gko::matrix_data<std::complex<double>, gko::int64> data{
        {{1.0, 0.0}, {2.0, 3.2}},
        {{2.1, 2.2}, {0.0, 0.0}},
        {{0.0, 3.0}, {3.2, 5.3}}};
    // clang-format on
    std::ostringstream oss{};

    write_raw(oss, data);

    ASSERT_EQ(oss.str(),
              "%%MatrixMarket matrix array complex general\n"
              "3 2\n"
              "1 0\n"
              "2.1 2.2\n"
              "0 3\n"
              "2 3.2\n"
              "0 0\n"
              "3.2 5.3\n");
}


TEST(MatrixData, WritesComplexFloatMatrixToMatrixMarketCoordinateWith64Index)
{
    // clang-format off
    gko::matrix_data<std::complex<float>, gko::int64> data{
        {{1.0, 0.0}, {2.0, 3.2}},
        {{2.1, 2.2}, {0.0, 0.0}},
        {{0.0, 3.0}, {3.2, 5.3}}};
    // clang-format on
    std::ostringstream oss{};

    write_raw(oss, data, gko::layout_type::coordinate);

    ASSERT_EQ(oss.str(),
              "%%MatrixMarket matrix coordinate complex general\n"
              "3 2 5\n"
              "1 1 1 0\n"
              "1 2 2 3.2\n"
              "2 1 2.1 2.2\n"
              "3 1 0 3\n"
              "3 2 3.2 5.3\n");
}


template <typename ValueType, typename IndexType>
class DummyLinOp
    : public gko::EnableLinOp<DummyLinOp<ValueType, IndexType>>,
      public gko::EnableCreateMethod<DummyLinOp<ValueType, IndexType>>,
      public gko::ReadableFromMatrixData<ValueType, IndexType>,
      public gko::WritableToMatrixData<ValueType, IndexType> {
    friend class gko::EnablePolymorphicObject<DummyLinOp, gko::LinOp>;
    friend class gko::EnableCreateMethod<DummyLinOp>;

public:
    using value_type = ValueType;
    using index_type = IndexType;
    using mat_data = gko::matrix_data<ValueType, IndexType>;

    void read(const mat_data &data) override { data_ = data; }

    void write(mat_data &data) const override { data = data_; }

protected:
    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override {}

    void apply_impl(const gko::LinOp *alpha, const gko::LinOp *b,
                    const gko::LinOp *beta, gko::LinOp *x) const override
    {}

    explicit DummyLinOp(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<DummyLinOp>(exec)
    {}

public:
    mat_data data_;
};


template <typename ValueIndexType>
class RealDummyLinOpTest : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
};

TYPED_TEST_CASE(RealDummyLinOpTest, gko::test::RealValueIndexTypes);


TYPED_TEST(RealDummyLinOpTest, ReadsLinOpFromStream)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using tpl = typename gko::matrix_data<value_type, index_type>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array real general\n"
        "2 3\n"
        "1.0\n"
        "0.0\n"
        "3.0\n"
        "5.0\n"
        "2.0\n"
        "0.0\n");

    auto lin_op = gko::read<DummyLinOp<value_type, index_type>>(
        iss, gko::ReferenceExecutor::create());

    const auto &data = lin_op->data_;
    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    const auto &v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, 1.0));
    ASSERT_EQ(v[1], tpl(0, 1, 3.0));
    ASSERT_EQ(v[2], tpl(0, 2, 2.0));
    ASSERT_EQ(v[3], tpl(1, 0, 0.0));
    ASSERT_EQ(v[4], tpl(1, 1, 5.0));
    ASSERT_EQ(v[5], tpl(1, 2, 0.0));
}


TYPED_TEST(RealDummyLinOpTest, WritesLinOpToStream)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array real general\n"
        "2 3\n"
        "1.0\n"
        "0.0\n"
        "3.0\n"
        "5.0\n"
        "2.0\n"
        "0.0\n");
    auto lin_op = gko::read<DummyLinOp<value_type, index_type>>(
        iss, gko::ReferenceExecutor::create());
    std::ostringstream oss{};

    write(oss, lend(lin_op));

    ASSERT_EQ(oss.str(),
              "%%MatrixMarket matrix array real general\n"
              "2 3\n"
              "1\n"
              "0\n"
              "3\n"
              "5\n"
              "2\n"
              "0\n");
}


template <typename ValueIndexType>
class ComplexDummyLinOpTest : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
};

TYPED_TEST_CASE(ComplexDummyLinOpTest, gko::test::ComplexValueIndexTypes);


TYPED_TEST(ComplexDummyLinOpTest, ReadsLinOpFromStream)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using tpl = typename gko::matrix_data<value_type, index_type>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array complex general\n"
        "2 3\n"
        "1.0 2.0\n"
        "0.0 0.0\n"
        "3.0 4.0\n"
        "5.0 6.0\n"
        "2.0 3.0\n"
        "0.0 0.0\n");

    auto lin_op = gko::read<DummyLinOp<value_type, index_type>>(
        iss, gko::ReferenceExecutor::create());

    const auto &data = lin_op->data_;
    ASSERT_EQ(data.size, gko::dim<2>(2, 3));
    const auto &v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, value_type{1.0, 2.0}));
    ASSERT_EQ(v[1], tpl(0, 1, value_type{3.0, 4.0}));
    ASSERT_EQ(v[2], tpl(0, 2, value_type{2.0, 3.0}));
    ASSERT_EQ(v[3], tpl(1, 0, value_type{0.0, 0.0}));
    ASSERT_EQ(v[4], tpl(1, 1, value_type{5.0, 6.0}));
    ASSERT_EQ(v[5], tpl(1, 2, value_type{0.0, 0.0}));
}


TYPED_TEST(ComplexDummyLinOpTest, WritesLinOpToStream)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array complex general\n"
        "2 3\n"
        "1.0 2.0\n"
        "0.0 0.0\n"
        "3.0 4.0\n"
        "5.0 6.0\n"
        "2.0 3.0\n"
        "0.0 0.0\n");
    auto lin_op = gko::read<DummyLinOp<value_type, index_type>>(
        iss, gko::ReferenceExecutor::create());
    std::ostringstream oss{};

    write(oss, lend(lin_op));

    ASSERT_EQ(oss.str(),
              "%%MatrixMarket matrix array complex general\n"
              "2 3\n"
              "1 2\n"
              "0 0\n"
              "3 4\n"
              "5 6\n"
              "2 3\n"
              "0 0\n");
}


}  // namespace
