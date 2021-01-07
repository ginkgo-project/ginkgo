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

#include <ginkgo/core/matrix/diagonal.hpp>


#include <algorithm>
#include <complex>
#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/diagonal_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename ValueType>
class Diagonal : public ::testing::Test {
protected:
    using value_type = ValueType;
    using Csr = gko::matrix::Csr<value_type>;
    using Diag = gko::matrix::Diagonal<value_type>;
    using Dense = gko::matrix::Dense<value_type>;
    Diagonal()
        : exec(gko::ReferenceExecutor::create()),
          diag1(Diag::create(exec, 2)),
          diag2(Diag::create(exec, 3)),
          dense1(gko::initialize<Dense>(4, {{1.0, 2.0, 3.0}, {1.5, 2.5, 3.5}},
                                        exec)),
          dense2(gko::initialize<Dense>(4, {{1.0, 2.0, 3.0}, {1.5, 2.5, 3.5}},
                                        exec))
    {
        csr1 = Csr::create(exec);
        csr1->copy_from(dense1.get());
        csr2 = Csr::create(exec);
        csr2->copy_from(dense2.get());
        this->create_diag1(diag1.get());
        this->create_diag2(diag2.get());
    }

    void create_diag1(Diag *d)
    {
        auto *v = d->get_values();
        v[0] = 2.0;
        v[1] = 3.0;
    }

    void create_diag2(Diag *d)
    {
        auto *v = d->get_values();
        v[0] = 2.0;
        v[1] = 3.0;
        v[2] = 4.0;
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Csr> csr1;
    std::unique_ptr<Csr> csr2;
    std::unique_ptr<Diag> diag1;
    std::unique_ptr<Diag> diag2;
    std::unique_ptr<Dense> dense1;
    std::unique_ptr<Dense> dense2;
};

TYPED_TEST_SUITE(Diagonal, gko::test::ValueTypes);


TYPED_TEST(Diagonal, AppliesToDense)
{
    using value_type = typename TestFixture::value_type;
    this->diag1->apply(this->dense1.get(), this->dense2.get());

    EXPECT_EQ(this->dense2->at(0, 0), value_type{2.0});
    EXPECT_EQ(this->dense2->at(0, 1), value_type{4.0});
    EXPECT_EQ(this->dense2->at(0, 2), value_type{6.0});
    EXPECT_EQ(this->dense2->at(1, 0), value_type{4.5});
    EXPECT_EQ(this->dense2->at(1, 1), value_type{7.5});
    EXPECT_EQ(this->dense2->at(1, 2), value_type{10.5});
}


TYPED_TEST(Diagonal, RightAppliesToDense)
{
    using value_type = typename TestFixture::value_type;
    this->diag2->rapply(this->dense1.get(), this->dense2.get());

    EXPECT_EQ(this->dense2->at(0, 0), value_type{2.0});
    EXPECT_EQ(this->dense2->at(0, 1), value_type{6.0});
    EXPECT_EQ(this->dense2->at(0, 2), value_type{12.0});
    EXPECT_EQ(this->dense2->at(1, 0), value_type{3.0});
    EXPECT_EQ(this->dense2->at(1, 1), value_type{7.5});
    EXPECT_EQ(this->dense2->at(1, 2), value_type{14.0});
}


TYPED_TEST(Diagonal, ApplyToDenseFailsForWrongInnerDimensions)
{
    using value_type = typename TestFixture::value_type;
    auto result =
        gko::matrix::Dense<value_type>::create(this->exec, gko::dim<2>{3});

    // 3x3 times 2x3 = 3x3 --> mismatch for inner dimensions
    ASSERT_THROW(this->diag2->apply(this->dense1.get(), result.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Diagonal, ApplyToDenseFailsForWrongNumberOfRows)
{
    using value_type = typename TestFixture::value_type;
    auto result =
        gko::matrix::Dense<value_type>::create(this->exec, gko::dim<2>{3});

    // 2x2 times 2x3 = 3x3 --> mismatch for rows of diagonal and result
    ASSERT_THROW(this->diag1->apply(this->dense1.get(), result.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Diagonal, ApplyToDenseFailsForWrongNumberOfCols)
{
    using value_type = typename TestFixture::value_type;
    auto result =
        gko::matrix::Dense<value_type>::create(this->exec, gko::dim<2>{2});

    // 2x2 times 2x3 = 2x2 --> mismatch for cols of dense1 and result
    ASSERT_THROW(this->diag1->apply(this->dense1.get(), result.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Diagonal, RightApplyToDenseFailsForWrongInnerDimensions)
{
    using value_type = typename TestFixture::value_type;
    auto result =
        gko::matrix::Dense<value_type>::create(this->exec, gko::dim<2>{2});

    // 2x3 times 2x2 = 2x2 --> mismatch for inner DimensionMismatch
    ASSERT_THROW(this->diag1->rapply(this->dense1.get(), result.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Diagonal, RightApplyToDenseFailsForWrongNumberOfRows)
{
    using value_type = typename TestFixture::value_type;
    auto result =
        gko::matrix::Dense<value_type>::create(this->exec, gko::dim<2>{3});

    // 2x3 times 3x3 = 3x3 --> mismatch for rows of dense1 and result
    ASSERT_THROW(this->diag2->rapply(this->dense1.get(), result.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Diagonal, RightApplyToDenseFailsForWrongNumberOfCols)
{
    using value_type = typename TestFixture::value_type;
    auto result =
        gko::matrix::Dense<value_type>::create(this->exec, gko::dim<2>{2});

    // 2x3 times 3x3 = 2x2 --> mismatch for cols of diagonal and result
    ASSERT_THROW(this->diag2->rapply(this->dense1.get(), result.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Diagonal, AppliesToCsr)
{
    using value_type = typename TestFixture::value_type;
    this->diag1->apply(this->csr1.get(), this->csr2.get());

    const auto values = this->csr2->get_const_values();
    const auto row_ptrs = this->csr2->get_const_row_ptrs();
    const auto col_idxs = this->csr2->get_const_col_idxs();

    EXPECT_EQ(this->csr2->get_num_stored_elements(), 6);
    EXPECT_EQ(values[0], value_type{2.0});
    EXPECT_EQ(values[1], value_type{4.0});
    EXPECT_EQ(values[2], value_type{6.0});
    EXPECT_EQ(values[3], value_type{4.5});
    EXPECT_EQ(values[4], value_type{7.5});
    EXPECT_EQ(values[5], value_type{10.5});
    EXPECT_EQ(row_ptrs[0], 0);
    EXPECT_EQ(row_ptrs[1], 3);
    EXPECT_EQ(row_ptrs[2], 6);
    EXPECT_EQ(col_idxs[0], 0);
    EXPECT_EQ(col_idxs[1], 1);
    EXPECT_EQ(col_idxs[2], 2);
    EXPECT_EQ(col_idxs[3], 0);
    EXPECT_EQ(col_idxs[4], 1);
    EXPECT_EQ(col_idxs[5], 2);
}


TYPED_TEST(Diagonal, RightAppliesToCsr)
{
    using value_type = typename TestFixture::value_type;
    this->diag2->rapply(this->csr1.get(), this->csr2.get());

    const auto values = this->csr2->get_const_values();
    const auto row_ptrs = this->csr2->get_const_row_ptrs();
    const auto col_idxs = this->csr2->get_const_col_idxs();

    EXPECT_EQ(this->csr2->get_num_stored_elements(), 6);
    EXPECT_EQ(values[0], value_type{2.0});
    EXPECT_EQ(values[1], value_type{6.0});
    EXPECT_EQ(values[2], value_type{12.0});
    EXPECT_EQ(values[3], value_type{3.0});
    EXPECT_EQ(values[4], value_type{7.5});
    EXPECT_EQ(values[5], value_type{14.0});
    EXPECT_EQ(row_ptrs[0], 0);
    EXPECT_EQ(row_ptrs[1], 3);
    EXPECT_EQ(row_ptrs[2], 6);
    EXPECT_EQ(col_idxs[0], 0);
    EXPECT_EQ(col_idxs[1], 1);
    EXPECT_EQ(col_idxs[2], 2);
    EXPECT_EQ(col_idxs[3], 0);
    EXPECT_EQ(col_idxs[4], 1);
    EXPECT_EQ(col_idxs[5], 2);
}


TYPED_TEST(Diagonal, ApplyToCsrFailsForWrongInnerDimensions)
{
    using value_type = typename TestFixture::value_type;
    auto result =
        gko::matrix::Csr<value_type>::create(this->exec, gko::dim<2>{3});

    // 3x3 times 2x3 = 3x3 --> mismatch for inner dimensions
    ASSERT_THROW(this->diag2->apply(this->csr1.get(), result.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Diagonal, ApplyToCsrFailsForWrongNumberOfRows)
{
    using value_type = typename TestFixture::value_type;
    auto result =
        gko::matrix::Csr<value_type>::create(this->exec, gko::dim<2>{3});

    // 2x2 times 2x3 = 3x3 --> mismatch for rows of diagonal and result
    ASSERT_THROW(this->diag1->apply(this->csr1.get(), result.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Diagonal, ApplyToCsrFailsForWrongNumberOfCols)
{
    using value_type = typename TestFixture::value_type;
    auto result =
        gko::matrix::Csr<value_type>::create(this->exec, gko::dim<2>{2});

    // 2x2 times 2x3 = 2x2 --> mismatch for cols of csr1 and result
    ASSERT_THROW(this->diag1->apply(this->csr1.get(), result.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Diagonal, RightApplyToCsrFailsForWrongInnerDimensions)
{
    using value_type = typename TestFixture::value_type;
    auto result =
        gko::matrix::Csr<value_type>::create(this->exec, gko::dim<2>{2});

    // 2x3 times 2x2 = 2x2 --> mismatch for inner DimensionMismatch
    ASSERT_THROW(this->diag1->rapply(this->csr1.get(), result.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Diagonal, RightApplyToCsrFailsForWrongNumberOfRows)
{
    using value_type = typename TestFixture::value_type;
    auto result =
        gko::matrix::Csr<value_type>::create(this->exec, gko::dim<2>{3});

    // 2x3 times 3x3 = 3x3 --> mismatch for rows of csr1 and result
    ASSERT_THROW(this->diag2->rapply(this->csr1.get(), result.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Diagonal, RightApplyToCsrFailsForWrongNumberOfCols)
{
    using value_type = typename TestFixture::value_type;
    auto result =
        gko::matrix::Csr<value_type>::create(this->exec, gko::dim<2>{2});

    // 2x3 times 3x3 = 2x2 --> mismatch for cols of diagonal and result
    ASSERT_THROW(this->diag2->rapply(this->csr1.get(), result.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Diagonal, ConvertsToCsr)
{
    using value_type = typename TestFixture::value_type;

    this->diag1->convert_to(this->csr1.get());

    const auto nnz = this->csr1->get_num_stored_elements();
    const auto row_ptrs = this->csr1->get_const_row_ptrs();
    const auto col_idxs = this->csr1->get_const_col_idxs();
    const auto values = this->csr1->get_const_values();

    EXPECT_EQ(nnz, 2);
    EXPECT_EQ(row_ptrs[0], 0);
    EXPECT_EQ(row_ptrs[1], 1);
    EXPECT_EQ(row_ptrs[2], 2);
    EXPECT_EQ(col_idxs[0], 0);
    EXPECT_EQ(col_idxs[1], 1);
    EXPECT_EQ(values[0], value_type(2.0));
    EXPECT_EQ(values[1], value_type(3.0));
}


TYPED_TEST(Diagonal, InplaceAbsolute)
{
    using value_type = typename TestFixture::value_type;

    this->diag1->compute_absolute_inplace();
    auto values = this->diag1->get_values();

    EXPECT_EQ(values[0], value_type(2.0));
    EXPECT_EQ(values[1], value_type(3.0));
}


TYPED_TEST(Diagonal, OutplaceAbsolute)
{
    using value_type = typename TestFixture::value_type;
    using abs_type = gko::remove_complex<value_type>;

    auto abs_diag = this->diag1->compute_absolute();
    auto values = abs_diag->get_values();

    EXPECT_EQ(values[0], value_type(2.0));
    EXPECT_EQ(values[1], value_type(3.0));
}


template <typename ValueType>
class DiagonalComplex : public ::testing::Test {
protected:
    using value_type = ValueType;
    using Diag = gko::matrix::Diagonal<value_type>;
};

TYPED_TEST_SUITE(DiagonalComplex, gko::test::ComplexValueTypes);


TYPED_TEST(DiagonalComplex, MtxIsConjugateTransposable)
{
    using Diag = typename TestFixture::Diag;
    using value_type = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto diag = Diag::create(exec, 3);
    auto diag_values = diag->get_values();
    diag_values[0] = value_type{1.0, 2.0};
    diag_values[1] = value_type{3.0, 0.0};
    diag_values[2] = value_type{0.0, 1.5};

    auto trans = diag->conj_transpose();
    auto trans_as_diagonal = static_cast<Diag *>(trans.get());
    auto trans_values = trans_as_diagonal->get_values();

    EXPECT_EQ(trans->get_size(), gko::dim<2>(3));
    EXPECT_EQ(trans_values[0], (value_type{1.0, -2.0}));
    EXPECT_EQ(trans_values[1], (value_type{3.0, 0.0}));
    EXPECT_EQ(trans_values[2], (value_type{0.0, -1.5}));
}


TYPED_TEST(DiagonalComplex, InplaceAbsolute)
{
    using Diag = typename TestFixture::Diag;
    using value_type = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto diag = Diag::create(exec, 3);
    auto diag_values = diag->get_values();
    diag_values[0] = value_type{3.0, -4.0};
    diag_values[1] = value_type{-3.0, 0.0};
    diag_values[2] = value_type{0.0, -1.5};

    diag->compute_absolute_inplace();

    EXPECT_EQ(diag_values[0], (value_type{5.0, 0.0}));
    EXPECT_EQ(diag_values[1], (value_type{3.0, 0.0}));
    EXPECT_EQ(diag_values[2], (value_type{1.5, 0.0}));
}


TYPED_TEST(DiagonalComplex, OutplaceAbsolute)
{
    using Diag = typename TestFixture::Diag;
    using value_type = typename TestFixture::value_type;
    using abs_type = gko::remove_complex<value_type>;
    auto exec = gko::ReferenceExecutor::create();
    auto diag = Diag::create(exec, 3);
    auto diag_values = diag->get_values();
    diag_values[0] = value_type{3.0, -4.0};
    diag_values[1] = value_type{-3.0, 0.0};
    diag_values[2] = value_type{0.0, -1.5};

    auto abs_diag = diag->compute_absolute();
    auto abs_values = abs_diag->get_values();

    EXPECT_EQ(abs_values[0], (value_type{5.0}));
    EXPECT_EQ(abs_values[1], (value_type{3.0}));
    EXPECT_EQ(abs_values[2], (value_type{1.5}));
}


}  // namespace
