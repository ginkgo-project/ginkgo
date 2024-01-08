// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/utils/matrix_utils.hpp"


#include <cmath>
#include <random>
#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/csr.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/matrix_generator.hpp"


namespace {


template <typename T>
class MatrixUtils : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<T>;
    using mtx_type = gko::matrix::Dense<T>;
    using mtx_data = gko::matrix_data<value_type, int>;

    MatrixUtils()
        : exec(gko::ReferenceExecutor::create()),
          data(gko::test::generate_random_matrix_data<value_type, int>(
              500, 500, std::normal_distribution<real_type>(50, 5),
              std::normal_distribution<real_type>(20.0, 5.0),
              std::default_random_engine(42))),
          rectangular_data(gko::dim<2>(500, 100))
    {}

    std::shared_ptr<const gko::Executor> exec;
    mtx_data data;
    mtx_data rectangular_data;
};

TYPED_TEST_SUITE(MatrixUtils, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(MatrixUtils, MakeSymmetricThrowsError)
{
    ASSERT_THROW(gko::utils::make_symmetric(this->rectangular_data),
                 gko::DimensionMismatch);
}

TYPED_TEST(MatrixUtils, MakeHermitianThrowsError)
{
    ASSERT_THROW(gko::utils::make_hermitian(this->rectangular_data),
                 gko::DimensionMismatch);
}


TYPED_TEST(MatrixUtils, MakeDiagDominantThrowsError)
{
    ASSERT_THROW(gko::utils::make_diag_dominant(this->data, 0.9),
                 gko::ValueMismatch);
}


TYPED_TEST(MatrixUtils, MakeHpdMatrixThrowsError)
{
    ASSERT_THROW(gko::utils::make_hpd(this->data, 1.0), gko::ValueMismatch);
}


TYPED_TEST(MatrixUtils, MakeLowerTriangularCorrectly)
{
    auto orig_mtx = TestFixture::mtx_type::create(this->exec);
    orig_mtx->read(this->data);
    gko::utils::make_lower_triangular(this->data);

    auto mtx = TestFixture::mtx_type::create(this->exec);
    mtx->read(this->data);
    for (gko::size_type i = 0; i < mtx->get_size()[0]; i++) {
        for (gko::size_type j = 0; j <= i; j++) {
            ASSERT_EQ(mtx->at(i, j), orig_mtx->at(i, j));
        }
        for (gko::size_type j = i + 1; j < mtx->get_size()[1]; j++) {
            ASSERT_EQ(mtx->at(i, j), gko::zero<TypeParam>());
        }
    }
}


TYPED_TEST(MatrixUtils, MakeUpperTriangularCorrectly)
{
    auto orig_mtx = TestFixture::mtx_type::create(this->exec);
    orig_mtx->read(this->data);
    gko::utils::make_upper_triangular(this->data);

    auto mtx = TestFixture::mtx_type::create(this->exec);
    mtx->read(this->data);
    for (gko::size_type i = 0; i < mtx->get_size()[0]; i++) {
        for (gko::size_type j = 0; j < i; j++) {
            ASSERT_EQ(mtx->at(i, j), gko::zero<TypeParam>());
        }
        for (gko::size_type j = i; j < mtx->get_size()[1]; j++) {
            ASSERT_EQ(mtx->at(i, j), orig_mtx->at(i, j));
        }
    }
}


TYPED_TEST(MatrixUtils, MakeRemoveDiagonalCorrectly)
{
    gko::utils::make_remove_diagonal(this->data);

    auto mtx = TestFixture::mtx_type::create(this->exec);
    mtx->read(this->data);
    for (gko::size_type i = 0;
         i < std::min(mtx->get_size()[0], mtx->get_size()[1]); i++) {
        ASSERT_EQ(mtx->at(i, i), gko::zero<TypeParam>());
    }
}


TYPED_TEST(MatrixUtils, MakeUnitDiagonalCorrectly)
{
    gko::utils::make_unit_diagonal(this->data);

    auto mtx = TestFixture::mtx_type::create(this->exec);
    mtx->read(this->data);
    for (gko::size_type i = 0;
         i < std::min(mtx->get_size()[0], mtx->get_size()[1]); i++) {
        ASSERT_EQ(mtx->at(i, i), gko::one<TypeParam>());
    }
}


TYPED_TEST(MatrixUtils, MakeSymmetricCorrectly)
{
    gko::utils::make_symmetric(this->data);

    auto mtx = TestFixture::mtx_type::create(this->exec);
    mtx->read(this->data);
    for (gko::size_type i = 0; i < mtx->get_size()[0]; i++) {
        for (gko::size_type j = 0; j <= i; j++) {
            ASSERT_EQ(mtx->at(i, j), mtx->at(j, i));
        }
    }
}


TYPED_TEST(MatrixUtils, MakeHermitianCorrectly)
{
    gko::utils::make_hermitian(this->data);

    auto mtx = TestFixture::mtx_type::create(this->exec);
    mtx->read(this->data);
    for (gko::size_type i = 0; i < mtx->get_size()[0]; i++) {
        for (gko::size_type j = 0; j <= i; j++) {
            ASSERT_EQ(mtx->at(i, j), gko::conj(mtx->at(j, i)));
        }
    }
}


TYPED_TEST(MatrixUtils, MakeDiagDominantCorrectly)
{
    using T = typename TestFixture::value_type;

    gko::utils::make_diag_dominant(this->data);

    auto mtx = TestFixture::mtx_type::create(this->exec);
    mtx->read(this->data);
    for (gko::size_type i = 0; i < mtx->get_size()[0]; i++) {
        gko::remove_complex<T> off_diag_abs = 0;
        for (gko::size_type j = 0; j < mtx->get_size()[1]; j++) {
            if (j != i) {
                off_diag_abs += gko::abs(mtx->at(i, j));
            }
        }
        ASSERT_GT(gko::abs(mtx->at(i, i)) * (1 + r<T>::value), off_diag_abs);
    }
}


TYPED_TEST(MatrixUtils, MakeDiagDominantWithRatioCorrectly)
{
    using T = typename TestFixture::value_type;
    gko::remove_complex<T> ratio = 1.001;

    gko::utils::make_diag_dominant(this->data, ratio);

    auto mtx = TestFixture::mtx_type::create(this->exec);
    mtx->read(this->data);
    for (gko::size_type i = 0; i < mtx->get_size()[0]; i++) {
        gko::remove_complex<T> off_diag_abs = 0;
        for (gko::size_type j = 0; j < mtx->get_size()[1]; j++) {
            if (j != i) {
                off_diag_abs += gko::abs(mtx->at(i, j));
            }
        }
        ASSERT_GT(gko::abs(mtx->at(i, i)) * (1 + r<T>::value),
                  off_diag_abs * ratio);
    }
}


TYPED_TEST(MatrixUtils, MakeDiagDominantWithEmptyOffdiagRowCorrectly)
{
    using value_type = typename TestFixture::value_type;
    using entry = gko::matrix_data_entry<value_type, int>;
    gko::matrix_data<value_type, int> data{gko::dim<2>{3, 3}};
    data.nonzeros.emplace_back(0, 0, gko::one<value_type>());
    data.nonzeros.emplace_back(1, 1, gko::zero<value_type>());

    gko::utils::make_diag_dominant(data, 1.0);

    ASSERT_EQ(data.nonzeros.size(), 3);
    ASSERT_EQ(data.nonzeros[0], (entry{0, 0, gko::one<value_type>()}));
    ASSERT_EQ(data.nonzeros[1], (entry{1, 1, gko::one<value_type>()}));
    ASSERT_EQ(data.nonzeros[2], (entry{2, 2, gko::one<value_type>()}));
}


TYPED_TEST(MatrixUtils, MakeHpdMatrixCorrectly)
{
    using T = typename TestFixture::value_type;
    auto cpy_data = this->data;

    gko::utils::make_hpd(this->data, 1.001);
    gko::utils::make_hermitian(cpy_data);
    gko::utils::make_diag_dominant(cpy_data, 1.001);

    auto mtx = TestFixture::mtx_type::create(this->exec);
    mtx->read(this->data);
    auto cpy_mtx = TestFixture::mtx_type::create(this->exec);
    cpy_mtx->read(cpy_data);
    GKO_ASSERT_MTX_NEAR(mtx, cpy_mtx, r<T>::value);
}


TYPED_TEST(MatrixUtils, MakeHpdMatrixWithRatioCorrectly)
{
    using T = typename TestFixture::value_type;
    gko::remove_complex<T> ratio = 1.00001;
    auto cpy_data = this->data;

    gko::utils::make_hpd(this->data, ratio);
    gko::utils::make_hermitian(cpy_data);
    gko::utils::make_diag_dominant(cpy_data, ratio);

    auto mtx = TestFixture::mtx_type::create(this->exec);
    mtx->read(this->data);
    auto cpy_mtx = TestFixture::mtx_type::create(this->exec);
    cpy_mtx->read(cpy_data);
    GKO_ASSERT_MTX_NEAR(mtx, cpy_mtx, r<T>::value);
}


TYPED_TEST(MatrixUtils, MakeSpdMatrixCorrectly)
{
    using T = typename TestFixture::value_type;
    auto cpy_data = this->data;

    gko::utils::make_spd(this->data, 1.001);
    gko::utils::make_symmetric(cpy_data);
    gko::utils::make_diag_dominant(cpy_data, 1.001);

    auto mtx = TestFixture::mtx_type::create(this->exec);
    mtx->read(this->data);
    auto cpy_mtx = TestFixture::mtx_type::create(this->exec);
    cpy_mtx->read(cpy_data);
    GKO_ASSERT_MTX_NEAR(mtx, cpy_mtx, r<T>::value);
}


TYPED_TEST(MatrixUtils, MakeSpdMatrixWithRatioCorrectly)
{
    using T = typename TestFixture::value_type;
    gko::remove_complex<T> ratio = 1.00001;
    auto cpy_data = this->data;

    gko::utils::make_spd(this->data, ratio);
    gko::utils::make_symmetric(cpy_data);
    gko::utils::make_diag_dominant(cpy_data, ratio);

    auto mtx = TestFixture::mtx_type::create(this->exec);
    mtx->read(this->data);
    auto cpy_mtx = TestFixture::mtx_type::create(this->exec);
    cpy_mtx->read(cpy_data);
    GKO_ASSERT_MTX_NEAR(mtx, cpy_mtx, r<T>::value);
}


TEST(MatrixUtils, RemoveDiagonalEntry)
{
    using T = float;
    using Csr = gko::matrix::Csr<T, int>;
    auto exec = gko::ReferenceExecutor::create();
    auto b = gko::initialize<Csr>(
        {I<T>{2.0, 0.0, 1.1, 0.0}, I<T>{1.0, 2.4, 0.0, -1.0},
         I<T>{0.0, -4.0, 2.2, -2.0}, I<T>{0.0, -3.0, 1.5, 1.0}},
        exec);
    const int row_to_remove = 2;

    gko::utils::remove_diagonal_entry_from_row(b.get(), row_to_remove);

    const auto rowptrs = b->get_const_row_ptrs();
    const auto colidxs = b->get_const_col_idxs();
    for (int i = 0; i < 4; i++) {
        bool has_diag = false;
        for (int j = rowptrs[i]; j < rowptrs[i + 1]; j++) {
            if (colidxs[j] == i) {
                has_diag = true;
            }
        }
        ASSERT_EQ(has_diag, i != row_to_remove);
    }
}


TEST(MatrixUtils, ModifyToEnsureAllDiagonalEntries)
{
    using T = float;
    using Csr = gko::matrix::Csr<T, int>;
    auto exec = gko::ReferenceExecutor::create();
    auto check_all_diag = [](const Csr* csr) {
        const auto rowptrs = csr->get_const_row_ptrs();
        const auto colidxs = csr->get_const_col_idxs();
        const auto ndiag =
            static_cast<int>(std::min(csr->get_size()[0], csr->get_size()[1]));
        bool all_diags = true;
        for (int i = 0; i < ndiag; i++) {
            bool has_diag = false;
            for (int j = rowptrs[i]; j < rowptrs[i + 1]; j++) {
                if (colidxs[j] == i) {
                    has_diag = true;
                    break;
                }
            }
            if (!has_diag) {
                all_diags = false;
                break;
            }
        }
        return all_diags;
    };
    auto b = gko::initialize<Csr>({I<T>{2.0, 0.0, 1.1}, I<T>{1.0, 0.0, 0.0},
                                   I<T>{0.0, -4.0, 2.2}, I<T>{0.0, -3.0, 1.5}},
                                  exec);
    // ensure it misses some diag
    bool prev_check = check_all_diag(b.get());

    gko::utils::ensure_all_diagonal_entries(b.get());

    ASSERT_FALSE(prev_check);
    ASSERT_TRUE(check_all_diag(b.get()));
}


}  // namespace
