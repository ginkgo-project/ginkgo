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

#include "core/test/utils/matrix_utils.hpp"


#include <cmath>
#include <random>
#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/csr.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/matrix_generator.hpp"
#include "core/test/utils/matrix_utils.hpp"


namespace {


template <typename T>
class MatrixUtils : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<T>;
    using mtx_type = gko::matrix::Dense<T>;

    MatrixUtils()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::test::generate_random_matrix<mtx_type>(
              500, 500, std::normal_distribution<real_type>(50, 5),
              std::normal_distribution<real_type>(20.0, 5.0),
              std::default_random_engine(42), exec)),
          unsquare_mtx(mtx_type::create(exec, gko::dim<2>(500, 100)))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<mtx_type> mtx;
    std::unique_ptr<mtx_type> unsquare_mtx;
};

TYPED_TEST_SUITE(MatrixUtils, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(MatrixUtils, MakeSymmetricThrowsError)
{
    ASSERT_THROW(gko::test::make_symmetric(gko::lend(this->unsquare_mtx)),
                 gko::DimensionMismatch);
}

TYPED_TEST(MatrixUtils, MakeHermitianThrowsError)
{
    ASSERT_THROW(gko::test::make_hermitian(gko::lend(this->unsquare_mtx)),
                 gko::DimensionMismatch);
}


TYPED_TEST(MatrixUtils, MakeDiagDominantThrowsError)
{
    ASSERT_THROW(gko::test::make_diag_dominant(gko::lend(this->mtx), 0.9),
                 gko::ValueMismatch);
}


TYPED_TEST(MatrixUtils, MakeHpdMatrixThrowsError)
{
    ASSERT_THROW(gko::test::make_hpd(gko::lend(this->mtx), 1.0),
                 gko::ValueMismatch);
}


TYPED_TEST(MatrixUtils, MakeSymmetricCorrectly)
{
    gko::test::make_symmetric(gko::lend(this->mtx));

    for (gko::size_type i = 0; i < this->mtx->get_size()[0]; i++) {
        for (gko::size_type j = 0; j <= i; j++) {
            ASSERT_EQ(this->mtx->at(i, j), this->mtx->at(j, i));
        }
    }
}


TYPED_TEST(MatrixUtils, MakeHermitianCorrectly)
{
    gko::test::make_hermitian(gko::lend(this->mtx));

    for (gko::size_type i = 0; i < this->mtx->get_size()[0]; i++) {
        for (gko::size_type j = 0; j <= i; j++) {
            ASSERT_EQ(this->mtx->at(i, j), gko::conj(this->mtx->at(j, i)));
        }
    }
}


TYPED_TEST(MatrixUtils, MakeDiagDominantCorrectly)
{
    using T = typename TestFixture::value_type;
    // make_diag_dominant also consider diag value.
    // To check the ratio easily, set the diag zeros
    for (gko::size_type i = 0; i < this->mtx->get_size()[0]; i++) {
        this->mtx->at(i, i) = 0;
    }

    gko::test::make_diag_dominant(gko::lend(this->mtx));

    for (gko::size_type i = 0; i < this->mtx->get_size()[0]; i++) {
        gko::remove_complex<T> off_diag_abs = 0;
        for (gko::size_type j = 0; j < this->mtx->get_size()[1]; j++) {
            if (j != i) {
                off_diag_abs += std::abs(this->mtx->at(i, j));
            }
        }
        ASSERT_NEAR(gko::real(this->mtx->at(i, i)), off_diag_abs, r<T>::value);
    }
}


TYPED_TEST(MatrixUtils, MakeDiagDominantWithRatioCorrectly)
{
    using T = typename TestFixture::value_type;
    gko::remove_complex<T> ratio = 1.001;
    // make_diag_dominant also consider diag value.
    // To check the ratio easily, set the diag zeros
    for (gko::size_type i = 0; i < this->mtx->get_size()[0]; i++) {
        this->mtx->at(i, i) = 0;
    }

    gko::test::make_diag_dominant(gko::lend(this->mtx), ratio);

    for (gko::size_type i = 0; i < this->mtx->get_size()[0]; i++) {
        gko::remove_complex<T> off_diag_abs = 0;
        for (gko::size_type j = 0; j < this->mtx->get_size()[1]; j++) {
            if (j != i) {
                off_diag_abs += std::abs(this->mtx->at(i, j));
            }
        }
        ASSERT_NEAR(gko::real(this->mtx->at(i, i)), off_diag_abs * ratio,
                    r<T>::value);
    }
}


TYPED_TEST(MatrixUtils, MakeHpdMatrixCorrectly)
{
    using T = typename TestFixture::value_type;
    auto cpy_mtx = this->mtx->clone();

    gko::test::make_hpd(gko::lend(this->mtx));
    gko::test::make_hermitian(gko::lend(cpy_mtx));
    gko::test::make_diag_dominant(gko::lend(cpy_mtx), 1.001);

    GKO_ASSERT_MTX_NEAR(this->mtx, cpy_mtx, r<T>::value);
}


TYPED_TEST(MatrixUtils, MakeHpdMatrixWithRatioCorrectly)
{
    using T = typename TestFixture::value_type;
    gko::remove_complex<T> ratio = 1.00001;
    auto cpy_mtx = this->mtx->clone();

    gko::test::make_hpd(gko::lend(this->mtx), ratio);
    gko::test::make_hermitian(gko::lend(cpy_mtx));
    gko::test::make_diag_dominant(gko::lend(cpy_mtx), ratio);

    GKO_ASSERT_MTX_NEAR(this->mtx, cpy_mtx, r<T>::value);
}


TEST(MatrixUtils, ModifyToEnsureAllDiagonalEntries)
{
    using T = float;
    using Csr = gko::matrix::Csr<T, int>;
    auto exec = gko::ReferenceExecutor::create();
    auto b = gko::initialize<Csr>(
        {I<T>{2.0, 0.0, 1.1, 0.0}, I<T>{1.0, 2.4, 0.0, -1.0},
         I<T>{0.0, -4.0, 2.2, -2.0}, I<T>{0.0, -3.0, 1.5, 1.0}},
        exec);

    gko::test::modify_to_ensure_all_diagonal_entries(b.get());

    const auto rowptrs = b->get_const_row_ptrs();
    const auto colidxs = b->get_const_col_idxs();
    bool all_diags = true;
    for (int i = 0; i < 3; i++) {
        bool has_diag = false;
        for (int j = rowptrs[i]; j < rowptrs[i + 1]; j++) {
            if (colidxs[j] == i) {
                has_diag = true;
            }
        }
        if (!has_diag) {
            all_diags = false;
            break;
        }
    }
    ASSERT_TRUE(all_diags);
}


}  // namespace
