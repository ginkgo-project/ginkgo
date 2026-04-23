// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/test/utils/matrix_generator.hpp"

#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include "core/base/utils.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename T>
class MatrixGenerator : public ::testing::Test {
protected:
    using value_type = T;
    using check_type = double;
    using real_type = gko::remove_complex<T>;
    using mtx_type = gko::matrix::Dense<T>;

    MatrixGenerator()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::test::generate_random_matrix<mtx_type>(
              500, 100, std::normal_distribution<>(50, 5),
              std::normal_distribution<>(20.0, 5.0),
              std::default_random_engine(42), exec)),
          dense_mtx(gko::test::generate_random_dense_matrix<value_type>(
              500, 100, std::normal_distribution<>(20.0, 5.0),
              std::default_random_engine(41), exec)),
          l_mtx(gko::test::generate_random_lower_triangular_matrix<mtx_type>(
              4, true, std::normal_distribution<>(50, 5),
              std::normal_distribution<>(20.0, 5.0),
              std::default_random_engine(42), exec)),
          u_mtx(gko::test::generate_random_upper_triangular_matrix<mtx_type>(
              4, true, std::normal_distribution<>(50, 5),
              std::normal_distribution<>(20.0, 5.0),
              std::default_random_engine(42), exec)),
          lower_bandwidth(2),
          upper_bandwidth(3),
          band_mtx(gko::test::generate_random_band_matrix<mtx_type>(
              100, lower_bandwidth, upper_bandwidth,
              std::normal_distribution<>(20.0, 5.0),
              std::default_random_engine(42), exec)),
          nnz_per_row_sample(500, 0),
          values_sample(0),
          band_values_sample(0)
    {
        // collect samples of nnz/row and values from the matrix
        for (int row = 0; row < mtx->get_size()[0]; ++row) {
            for (int col = 0; col < mtx->get_size()[1]; ++col) {
                auto val = mtx->at(row, col);
                if (val != gko::zero<T>()) {
                    ++nnz_per_row_sample[row];
                    values_sample.push_back(val);
                }
            }
        }

        // collect samples of nnz/row and values from the dense matrix
        for (int row = 0; row < dense_mtx->get_size()[0]; ++row) {
            for (int col = 0; col < dense_mtx->get_size()[1]; ++col) {
                auto val = dense_mtx->at(row, col);
                dense_values_sample.push_back(val);
            }
        }

        // collect samples of values from the band matrix
        for (int row = 0; row < band_mtx->get_size()[0]; ++row) {
            for (int col = 0; col < band_mtx->get_size()[1]; ++col) {
                auto val = band_mtx->at(row, col);
                if ((col - row <= upper_bandwidth) &&
                    (row - col <= lower_bandwidth)) {
                    band_values_sample.push_back(val);
                }
            }
        }
    }

    std::shared_ptr<const gko::Executor> exec;
    int lower_bandwidth;
    int upper_bandwidth;
    std::unique_ptr<mtx_type> mtx;
    std::unique_ptr<mtx_type> dense_mtx;
    std::unique_ptr<mtx_type> l_mtx;
    std::unique_ptr<mtx_type> u_mtx;
    std::unique_ptr<mtx_type> band_mtx;
    std::vector<int> nnz_per_row_sample;
    std::vector<T> values_sample;
    std::vector<T> dense_values_sample;
    std::vector<T> band_values_sample;


    template <typename InputIterator, typename ValueType, typename Closure>
    check_type get_nth_moment(int n, ValueType c, InputIterator sample_start,
                              InputIterator sample_end, Closure closure_op)
    {
        using std::pow;
        check_type res = 0;
        check_type num_elems = 0;
        while (sample_start != sample_end) {
            auto tmp = *(sample_start++);
            res += pow(static_cast<check_type>(closure_op(tmp)) -
                           static_cast<check_type>(c),
                       n);
            num_elems += 1;
        }
        return res / num_elems;
    }

    template <typename ValueType, typename InputIterator, typename Closure>
    void check_average_and_deviation(
        InputIterator sample_start, InputIterator sample_end,
        gko::remove_complex<ValueType> average_ans,
        gko::remove_complex<ValueType> deviation_ans, Closure closure_op)
    {
        auto average =
            this->get_nth_moment(1, gko::zero<gko::remove_complex<ValueType>>(),
                                 sample_start, sample_end, closure_op);
        auto deviation = sqrt(this->get_nth_moment(2, average, sample_start,
                                                   sample_end, closure_op));

        // check that average & deviation is within 10% of the required amount
        ASSERT_NEAR(average, average_ans, average_ans * 0.1);
        ASSERT_NEAR(deviation, deviation_ans, deviation_ans * 0.1);
    }
};

TYPED_TEST_SUITE(MatrixGenerator, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(MatrixGenerator, OutputHasCorrectSize)
{
    ASSERT_EQ(this->mtx->get_size(), gko::dim<2>(500, 100));
}


TYPED_TEST(MatrixGenerator, OutputHasCorrectNonzeroAverageAndDeviation)
{
    using T = typename TestFixture::value_type;
    // this test only tests integer distributions, so only test real types
    if (gko::is_complex<T>()) {
        GTEST_SKIP();
    }
    this->template check_average_and_deviation<T>(
        begin(this->nnz_per_row_sample), end(this->nnz_per_row_sample), 50.0,
        5.0, [](T val) { return gko::real(val); });
}


TYPED_TEST(MatrixGenerator, OutputHasCorrectValuesAverageAndDeviation)
{
    using T = typename TestFixture::value_type;
    // check the real part
    this->template check_average_and_deviation<T>(
        begin(this->values_sample), end(this->values_sample), 20.0, 5.0,
        [](T& val) { return gko::real(val); });
    // check the imag part when the type is complex
    if (!std::is_same<T, gko::remove_complex<T>>::value) {
        this->template check_average_and_deviation<T>(
            begin(this->values_sample), end(this->values_sample), 20.0, 5.0,
            [](T& val) { return gko::imag(val); });
    }
}


TYPED_TEST(MatrixGenerator, DenseOutputHasCorrectValuesAverageAndDeviation)
{
    using T = typename TestFixture::value_type;
    // check the real part
    this->template check_average_and_deviation<T>(
        begin(this->dense_values_sample), end(this->dense_values_sample), 20.0,
        5.0, [](T& val) { return gko::real(val); });
    // check the imag part when the type is complex
    if (!std::is_same<T, gko::remove_complex<T>>::value) {
        this->template check_average_and_deviation<T>(
            begin(this->dense_values_sample), end(this->dense_values_sample),
            20.0, 5.0, [](T& val) { return gko::imag(val); });
    }
}


TYPED_TEST(MatrixGenerator, CanGenerateLowerTriangularMatrixWithDiagonalOnes)
{
    using T = typename TestFixture::value_type;
    ASSERT_EQ(this->l_mtx->at(0, 0), T{1.0});
    ASSERT_EQ(this->l_mtx->at(0, 1), T{0.0});
    ASSERT_EQ(this->l_mtx->at(0, 2), T{0.0});
    ASSERT_NE(this->l_mtx->at(1, 0), T{0.0});
    ASSERT_EQ(this->l_mtx->at(1, 1), T{1.0});
    ASSERT_EQ(this->l_mtx->at(1, 2), T{0.0});
    ASSERT_NE(this->l_mtx->at(2, 0), T{0.0});
    ASSERT_NE(this->l_mtx->at(2, 1), T{0.0});
    ASSERT_EQ(this->l_mtx->at(2, 2), T{1.0});
    ASSERT_NE(this->l_mtx->at(3, 0), T{0.0});
    ASSERT_NE(this->l_mtx->at(3, 1), T{0.0});
    ASSERT_NE(this->l_mtx->at(3, 2), T{0.0});
}


TYPED_TEST(MatrixGenerator, CanGenerateUpperTriangularMatrixWithDiagonalOnes)
{
    using T = typename TestFixture::value_type;
    ASSERT_EQ(this->u_mtx->at(0, 0), T{1.0});
    ASSERT_NE(this->u_mtx->at(0, 1), T{0.0});
    ASSERT_NE(this->u_mtx->at(0, 2), T{0.0});
    ASSERT_NE(this->u_mtx->at(0, 3), T{0.0});
    ASSERT_EQ(this->u_mtx->at(1, 0), T{0.0});
    ASSERT_EQ(this->u_mtx->at(1, 1), T{1.0});
    ASSERT_NE(this->u_mtx->at(1, 2), T{0.0});
    ASSERT_NE(this->u_mtx->at(1, 3), T{0.0});
    ASSERT_EQ(this->u_mtx->at(2, 0), T{0.0});
    ASSERT_EQ(this->u_mtx->at(2, 1), T{0.0});
    ASSERT_EQ(this->u_mtx->at(2, 2), T{1.0});
    ASSERT_NE(this->u_mtx->at(2, 3), T{0.0});
}


TYPED_TEST(MatrixGenerator, CanGenerateBandMatrix)
{
    using T = typename TestFixture::value_type;
    // the elements out of band are zero
    for (int row = 0; row < this->band_mtx->get_size()[0]; row++) {
        for (int col = 0; col < this->band_mtx->get_size()[1]; col++) {
            if ((col - row > this->upper_bandwidth) ||
                (row - col > this->lower_bandwidth)) {
                ASSERT_EQ(this->band_mtx->at(row, col), T{0.0});
            }
        }
    }
    // check the real part of elements in band
    this->template check_average_and_deviation<T>(
        begin(this->band_values_sample), end(this->band_values_sample), 20.0,
        5.0, [](T& val) { return gko::real(val); });
    // check the imag part when the type is complex
    if (!std::is_same<T, gko::remove_complex<T>>::value) {
        this->template check_average_and_deviation<T>(
            begin(this->band_values_sample), end(this->band_values_sample),
            20.0, 5.0, [](T& val) { return gko::imag(val); });
    }
}


TYPED_TEST(MatrixGenerator, CanGenerateTridiagMatrix)
{
    using T = typename TestFixture::value_type;
    using Dense = typename TestFixture::mtx_type;
    auto dist = std::normal_distribution<>(0, 1);
    auto engine = std::default_random_engine(42);
    auto lower = gko::test::detail::get_rand_value<T>(dist, engine);
    auto diag = gko::test::detail::get_rand_value<T>(dist, engine);
    auto upper = gko::test::detail::get_rand_value<T>(dist, engine);

    auto mtx = gko::test::generate_tridiag_matrix<Dense>(
        50, {lower, diag, upper}, this->exec);

    GKO_ASSERT_IS_SQUARE_MATRIX(mtx);
    for (gko::size_type i = 0; i < mtx->get_size()[0]; ++i) {
        ASSERT_EQ(mtx->at(i, i), diag);
        if (i > 0) {
            ASSERT_EQ(mtx->at(i, i - 1), lower);
            ASSERT_EQ(mtx->at(i - 1, i), upper);
        }
    }
}


TEST(MatrixGenerator, GeneratesLaplace2d5pointMatrixData)
{
    using T = std::complex<float>;
    using itype = long;
    using Dense = gko::matrix::Dense<T>;
    auto exec = gko::ReferenceExecutor::create();
    const gko::dim<2> dims{4, 4};
    auto diag = static_cast<T>(4.0);
    auto offdiag = static_cast<T>(-1.0);
    auto zero = gko::zero<T>();
    const gko::dim<2> matrix_dims{16, 16};
    const size_t nnz = 4 * 5 + 8 * 4 + 4 * 3;

    const auto ldata =
        gko::test::generate_laplacian_2d_5point_matrix_data<T, itype>(dims);
    auto dnlaplace = Dense::create(exec);
    dnlaplace->read(ldata);

    EXPECT_EQ(ldata.size, matrix_dims);
    EXPECT_EQ(ldata.nonzeros.size(), nnz);
    for (unsigned i = 0; i < matrix_dims[0]; i++) {
        EXPECT_EQ(dnlaplace->at(i, i), diag);
        int inz = 0;
        for (int j = 0; j < matrix_dims[1]; j++) {
            if (dnlaplace->at(i, j) != zero) {
                inz++;
                if (i != j) {
                    EXPECT_EQ(dnlaplace->at(i, j), offdiag);
                }
            }
        }
        EXPECT_GE(inz, 3);
        EXPECT_LE(inz, 5);
        // interior points
        if (i == 5 || i == 6 || i == 9 || i == 10) {
            EXPECT_EQ(inz, 5);
            for (int j = 0; j < matrix_dims[1]; j++) {
                if (j < i - 4 || j > i + 4) {
                    EXPECT_EQ(dnlaplace->at(i, 0), zero);
                }
            }
            EXPECT_EQ(dnlaplace->at(i, i + 1), offdiag);
            EXPECT_EQ(dnlaplace->at(i, i - 1), offdiag);
            EXPECT_EQ(dnlaplace->at(i, i - 4), offdiag);
            EXPECT_EQ(dnlaplace->at(i, i + 4), offdiag);
        }
    }
    // a corner point
    EXPECT_EQ(dnlaplace->at(15, 11), offdiag);
    EXPECT_EQ(dnlaplace->at(15, 14), offdiag);
    for (int j = 0; j < matrix_dims[1]; j++) {
        if (j != 14 && j != 11 && j != 15) {
            EXPECT_EQ(dnlaplace->at(15, j), zero);
        }
    }
}


TEST(MatrixGenerator, GeneratesLaplace3d27pointMatrixData)
{
    using T = float;
    using itype = int;
    using Dense = gko::matrix::Dense<T>;
    auto exec = gko::ReferenceExecutor::create();
    const gko::dim<3> dims{4, 4, 4};
    auto diag = static_cast<T>(26.0);
    auto offdiag = static_cast<T>(-1.0);
    auto zero = gko::zero<T>();
    // flat index: k*16 + j*4 + i

    const auto ldata =
        gko::test::generate_laplacian_3d_27point_matrix_data<T, itype>(dims);
    auto dnlaplace = Dense::create(exec);
    dnlaplace->read(ldata);

    ASSERT_EQ(ldata.size, (gko::dim<2>{64, 64}));
    // total nnz: each dim has 2 boundary (width 2) + 2 interior (width 3) = 10;
    // 10^3 = 1000
    ASSERT_EQ(ldata.nonzeros.size(), 1000u);

    // Corner point (0,0,0) = row 0: only +x/+y/+z quadrant → 7 neighbors + diag
    {
        int nz = 0;
        for (int col = 0; col < 64; col++) {
            if (dnlaplace->at(0, col) != zero) {
                nz++;
            }
        }
        EXPECT_EQ(nz, 8);
        EXPECT_EQ(dnlaplace->at(0, 0), diag);
        EXPECT_EQ(dnlaplace->at(0, 1), offdiag);   // (+1,0,0)
        EXPECT_EQ(dnlaplace->at(0, 4), offdiag);   // (0,+1,0)
        EXPECT_EQ(dnlaplace->at(0, 5), offdiag);   // (+1,+1,0)
        EXPECT_EQ(dnlaplace->at(0, 16), offdiag);  // (0,0,+1)
        EXPECT_EQ(dnlaplace->at(0, 17), offdiag);  // (+1,0,+1)
        EXPECT_EQ(dnlaplace->at(0, 20), offdiag);  // (0,+1,+1)
        EXPECT_EQ(dnlaplace->at(0, 21), offdiag);  // (+1,+1,+1)
    }

    // Face-center point (1,1,0) = row 5: k=0 removes one layer → 3*3*2 = 18
    // entries
    {
        int nz = 0;
        for (int col = 0; col < 64; col++) {
            if (dnlaplace->at(5, col) != zero) {
                nz++;
            }
        }
        EXPECT_EQ(nz, 18);
    }

    // Interior point (1,1,1) = row 21: full 3^3 stencil → 27 entries
    {
        int nz = 0;
        for (int col = 0; col < 64; col++) {
            if (dnlaplace->at(21, col) != zero) {
                nz++;
            }
        }
        EXPECT_EQ(nz, 27);
        for (int dk = -1; dk <= 1; dk++) {
            for (int dj = -1; dj <= 1; dj++) {
                for (int di = -1; di <= 1; di++) {
                    int col = (1 + dk) * 16 + (1 + dj) * 4 + (1 + di);
                    auto expected =
                        (di == 0 && dj == 0 && dk == 0) ? diag : offdiag;
                    EXPECT_EQ(dnlaplace->at(21, col), expected)
                        << "di=" << di << " dj=" << dj << " dk=" << dk;
                }
            }
        }
    }

    // Edge-center point (3,2,3) = row 59: i=3 and k=3 are max-boundary,
    // j=2 is interior → di in {-1,0}, dj in {-1,0,1}, dk in {-1,0}: 12 entries
    {
        int row = 59;  // 3*16 + 2*4 + 3
        int nz = 0;
        for (int col = 0; col < 64; col++) {
            if (dnlaplace->at(row, col) != zero) {
                nz++;
            }
        }
        EXPECT_EQ(nz, 12);
        EXPECT_EQ(dnlaplace->at(row, row), diag);
        EXPECT_EQ(dnlaplace->at(row, 58), offdiag);  // (-1,0,0): (2,2,3)
        EXPECT_EQ(dnlaplace->at(row, 55), offdiag);  // (0,-1,0): (3,1,3)
        EXPECT_EQ(dnlaplace->at(row, 63), offdiag);  // (0,+1,0): (3,3,3)
        EXPECT_EQ(dnlaplace->at(row, 43), offdiag);  // (0,0,-1): (3,2,2)
        EXPECT_EQ(dnlaplace->at(row, 38), offdiag);  // (-1,-1,-1): (2,1,2)
        EXPECT_EQ(dnlaplace->at(row, 46), offdiag);  // (-1,+1,-1): (2,3,2)
    }
}


}  // namespace
