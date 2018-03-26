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


#include <core/base/lin_op.hpp>


namespace {


TEST(MtxReader, ReadsDenseRealMtx)
{
    using tpl = gko::matrix_data<double, gko::int32>::nonzero_type;

    auto data = gko::read_raw<double, gko::int32>("data/dense_real.mtx");

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
    using tpl = gko::matrix_data<double, gko::int32>::nonzero_type;

    auto data = gko::read_raw<double, gko::int32>("data/dense_integer.mtx");

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
    using tpl = gko::matrix_data<cpx, gko::int32>::nonzero_type;

    auto data = gko::read_raw<cpx, gko::int32>("data/dense_complex.mtx");

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
    using tpl = gko::matrix_data<double, gko::int32>::nonzero_type;

    auto data = gko::read_raw<double, gko::int32>("data/sparse_real.mtx");

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
    using tpl = gko::matrix_data<double, gko::int32>::nonzero_type;

    auto data =
        gko::read_raw<double, gko::int32>("data/sparse_real_symmetric.mtx");

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
    using tpl = gko::matrix_data<double, gko::int32>::nonzero_type;

    auto data = gko::read_raw<double, gko::int32>(
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
    using tpl = gko::matrix_data<double, gko::int32>::nonzero_type;

    auto data = gko::read_raw<double, gko::int32>("data/sparse_pattern.mtx");

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
    using tpl = gko::matrix_data<cpx, gko::int32>::nonzero_type;

    auto data = gko::read_raw<cpx, gko::int32>("data/sparse_complex.mtx");

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
    using tpl = gko::matrix_data<cpx, gko::int32>::nonzero_type;

    auto data =
        gko::read_raw<cpx, gko::int32>("data/sparse_complex_hermitian.mtx");

    ASSERT_EQ(data.num_rows, 2);
    ASSERT_EQ(data.num_cols, 3);
    auto &v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 1, cpx(3.0, 1.0)));
    ASSERT_EQ(v[1], tpl(0, 2, cpx(2.0, 4.0)));
    ASSERT_EQ(v[2], tpl(1, 0, cpx(3.0, -1.0)));
    ASSERT_EQ(v[3], tpl(2, 0, cpx(2.0, -4.0)));
}


template <typename ValueType, typename IndexType>
class DummyLinOp : public gko::BasicLinOp<DummyLinOp<ValueType, IndexType>>,
                   public gko::ReadableFromMatrixData<ValueType, IndexType> {
    friend class gko::BasicLinOp<DummyLinOp>;

public:
    using value_type = ValueType;
    using index_type = IndexType;
    using mat_data = gko::matrix_data<ValueType, IndexType>;

    void apply(const gko::LinOp *b, gko::LinOp *x) const override {}

    void apply(const gko::LinOp *alpha, const gko::LinOp *b,
               const gko::LinOp *beta, gko::LinOp *x) const override
    {}

    void read(const mat_data &data) override { data_ = data; }

protected:
    explicit DummyLinOp(std::shared_ptr<const gko::Executor> exec)
        : gko::BasicLinOp<DummyLinOp>(exec, 0, 0, 0)
    {}

public:
    mat_data data_;
};


TEST(MtxReader, ReadsLinOpFromFile)
{
    using tpl = gko::matrix_data<double, gko::int32>::nonzero_type;

    auto lin_op = gko::read<DummyLinOp<double, gko::int32>>(
        "data/dense_real.mtx", gko::ReferenceExecutor::create());

    const auto &data = lin_op->data_;
    ASSERT_EQ(data.num_rows, 2);
    ASSERT_EQ(data.num_cols, 3);
    const auto &v = data.nonzeros;
    ASSERT_EQ(v[0], tpl(0, 0, 1.0));
    ASSERT_EQ(v[1], tpl(0, 1, 3.0));
    ASSERT_EQ(v[2], tpl(0, 2, 2.0));
    ASSERT_EQ(v[3], tpl(1, 0, 0.0));
    ASSERT_EQ(v[4], tpl(1, 1, 5.0));
    ASSERT_EQ(v[5], tpl(1, 2, 0.0));
}

}  // namespace
