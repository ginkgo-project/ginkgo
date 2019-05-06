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

#include <ginkgo/core/base/mtx_io.hpp>


#include <gtest/gtest.h>


#include <sstream>


#include <ginkgo/core/base/lin_op.hpp>


namespace {


TEST(MtxReader, ReadsDenseRealMtx)
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


TEST(MtxReader, ReadsDenseIntegerMtx)
{
    using tpl = gko::matrix_data<double, gko::int32>::nonzero_type;
    std::istringstream iss(
        "%%MatrixMarket matrix array integer general\n"
        "2 3\n"
        "1\n"
        "0\n"
        "3\n"
        "5\n"
        "2\n"
        "0\n");

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


TEST(MtxReader, ReadsDenseComplexMtx)
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


TEST(MatrixData, WritesRealMatrixToMatrixMarketArray)
{
    // clang-format off
    gko::matrix_data<> data{
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


TEST(MatrixData, WritesRealMatrixToMatrixMarketCoordinate)
{
    // clang-format off
    gko::matrix_data<> data{
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


TEST(MatrixData, WritesComplexMatrixToMatrixMarketArray)
{
    // clang-format off
    gko::matrix_data<std::complex<double>> data{
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


TEST(MatrixData, WritesComplexMatrixToMatrixMarketCoordinate)
{
    // clang-format off
    gko::matrix_data<std::complex<double>> data{
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


TEST(MtxReader, ReadsLinOpFromStream)
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

    auto lin_op = gko::read<DummyLinOp<double, gko::int32>>(
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


TEST(MtxReader, WritesLinOpToStream)
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
    auto lin_op = gko::read<DummyLinOp<double, gko::int32>>(
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


}  // namespace
