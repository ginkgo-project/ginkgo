// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/fft.hpp>


#include <complex>
#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"
#include "matrices/config.hpp"


namespace {


template <typename T>
class Fft : public ::testing::Test {
protected:
    using value_type = T;
    using Vec = gko::matrix::Dense<value_type>;
    using Mtx = gko::matrix::Fft;
    using Mtx2 = gko::matrix::Fft2;
    using Mtx3 = gko::matrix::Fft3;

    static value_type fourier_coef(gko::size_type n, gko::size_type i,
                                   gko::size_type j)
    {
        return gko::unit_root<value_type>(static_cast<int>(n),
                                          -static_cast<int>((i * j) % n));
    }

    Fft()
        : exec(gko::ReferenceExecutor::create()),
          rng{7381},
          n1{4},
          n2{8},
          n3{16},
          n{n1 * n2 * n3},
          nrhs{6},
          subcols{3},
          stride{8},
          size{n, nrhs},
          inv_n_scalar(gko::initialize<Vec>({T{1.f / n}}, exec)),
          frequency1{Vec::create(exec, size)},
          frequency2{Vec::create(exec, size)},
          frequency3{Vec::create(exec, size)},
          fft(Mtx::create(exec, n)),
          fft2(Mtx2::create(exec, n1 * n2, n3)),
          fft3(Mtx3::create(exec, n1, n2, n3)),
          dense_fft(Vec::create(exec, gko::dim<2>{n, n})),
          dense_fft2(Vec::create(exec, gko::dim<2>{n, n})),
          dense_fft3(Vec::create(exec, gko::dim<2>{n, n})),
          ifft(Mtx::create(exec, n, true)),
          ifft2(Mtx2::create(exec, n1 * n2, n3, true)),
          ifft3(Mtx3::create(exec, n1, n2, n3, true)),
          dense_ifft(Vec::create(exec, gko::dim<2>{n, n})),
          dense_ifft2(Vec::create(exec, gko::dim<2>{n, n})),
          dense_ifft3(Vec::create(exec, gko::dim<2>{n, n}))
    {
        std::uniform_int_distribution<gko::size_type> nz_dist(nrhs - 2, nrhs);
        std::uniform_real_distribution<gko::remove_complex<value_type>>
            val_dist(-1, 1);
        amplitude = gko::test::generate_random_matrix<Vec>(n, nrhs, nz_dist,
                                                           val_dist, rng, exec);
        for (gko::size_type i = 0; i < n; i++) {
            for (gko::size_type j = 0; j < n; j++) {
                dense_fft->at(i, j) = fourier_coef(n, i, j);
                dense_ifft->at(i, j) = conj(dense_fft->at(i, j));
            }
        }
        dense_fft->apply(amplitude, frequency1);
        auto idx2 = [&](gko::size_type x, gko::size_type y) {
            return x * n3 + y;
        };
        for (gko::size_type i1 = 0; i1 < n1 * n2; i1++) {
            for (gko::size_type i2 = 0; i2 < n3; i2++) {
                for (gko::size_type j1 = 0; j1 < n1 * n2; j1++) {
                    for (gko::size_type j2 = 0; j2 < n3; j2++) {
                        const auto i = idx2(i1, i2);
                        const auto j = idx2(j1, j2);
                        dense_fft2->at(i, j) = fourier_coef(n1 * n2, i1, j1) *
                                               fourier_coef(n3, i2, j2);
                        dense_ifft2->at(i, j) = conj(dense_fft2->at(i, j));
                    }
                }
            }
        }
        dense_fft2->apply(amplitude, frequency2);
        auto idx3 = [&](gko::size_type x, gko::size_type y, gko::size_type z) {
            return x * n2 * n3 + y * n3 + z;
        };
        for (gko::size_type i1 = 0; i1 < n1; i1++) {
            for (gko::size_type i2 = 0; i2 < n2; i2++) {
                for (gko::size_type i3 = 0; i3 < n3; i3++) {
                    for (gko::size_type j1 = 0; j1 < n1; j1++) {
                        for (gko::size_type j2 = 0; j2 < n2; j2++) {
                            for (gko::size_type j3 = 0; j3 < n3; j3++) {
                                const auto i = idx3(i1, i2, i3);
                                const auto j = idx3(j1, j2, j3);
                                dense_fft3->at(i, j) =
                                    fourier_coef(n1, i1, j1) *
                                    fourier_coef(n2, i2, j2) *
                                    fourier_coef(n3, i3, j3);
                                dense_ifft3->at(i, j) =
                                    conj(dense_fft3->at(i, j));
                            }
                        }
                    }
                }
            }
        }
        dense_fft3->apply(amplitude, frequency3);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::default_random_engine rng;
    gko::size_type n1;
    gko::size_type n2;
    gko::size_type n3;
    gko::size_type n;
    gko::size_type nrhs;
    gko::size_type subcols;
    gko::size_type stride;
    gko::dim<2> size;
    std::unique_ptr<Vec> inv_n_scalar;
    std::unique_ptr<Vec> amplitude;
    std::unique_ptr<Vec> frequency1;
    std::unique_ptr<Vec> frequency2;
    std::unique_ptr<Vec> frequency3;
    std::unique_ptr<Mtx> fft;
    std::unique_ptr<Mtx2> fft2;
    std::unique_ptr<Mtx3> fft3;
    std::unique_ptr<Vec> dense_fft;
    std::unique_ptr<Vec> dense_fft2;
    std::unique_ptr<Vec> dense_fft3;
    std::unique_ptr<Mtx> ifft;
    std::unique_ptr<Mtx2> ifft2;
    std::unique_ptr<Mtx3> ifft3;
    std::unique_ptr<Vec> dense_ifft;
    std::unique_ptr<Vec> dense_ifft2;
    std::unique_ptr<Vec> dense_ifft3;
};

TYPED_TEST_SUITE(Fft, gko::test::ComplexValueTypes, TypenameNameGenerator);


TYPED_TEST(Fft, ThrowsOnNonPowerOfTwo1D)
{
    auto wrong_fft = TestFixture::Mtx::create(this->exec, 3);
    auto wrong_vec = TestFixture::Vec::create(this->exec, gko::dim<2>(3, 1));

    ASSERT_THROW(wrong_fft->apply(wrong_vec, wrong_vec), gko::BadDimension);
}


TYPED_TEST(Fft, ThrowsOnNonPowerOfTwo2D)
{
    auto wrong_vec =
        TestFixture::Vec::create(this->exec, gko::dim<2>(3 * 2, 1));

    ASSERT_THROW(TestFixture::Mtx2::create(this->exec, 3, 2)
                     ->apply(wrong_vec, wrong_vec),
                 gko::BadDimension);
    ASSERT_THROW(TestFixture::Mtx2::create(this->exec, 2, 3)
                     ->apply(wrong_vec, wrong_vec),
                 gko::BadDimension);
}


TYPED_TEST(Fft, ThrowsOnNonPowerOfTwo3D)
{
    auto wrong_vec =
        TestFixture::Vec::create(this->exec, gko::dim<2>(3 * 2 * 4, 1));

    ASSERT_THROW(TestFixture::Mtx3::create(this->exec, 3, 2, 4)
                     ->apply(wrong_vec, wrong_vec),
                 gko::BadDimension);
    ASSERT_THROW(TestFixture::Mtx3::create(this->exec, 2, 3, 4)
                     ->apply(wrong_vec, wrong_vec),
                 gko::BadDimension);
    ASSERT_THROW(TestFixture::Mtx3::create(this->exec, 4, 2, 3)
                     ->apply(wrong_vec, wrong_vec),
                 gko::BadDimension);
}


TYPED_TEST(Fft, IsTransposable1D)
{
    auto transp = gko::as<typename TestFixture::Mtx>(this->fft->transpose());

    ASSERT_EQ(transp->get_size(), this->fft->get_size());
    ASSERT_FALSE(transp->is_inverse());
}


TYPED_TEST(Fft, InverseIsTransposable1D)
{
    auto transp = gko::as<typename TestFixture::Mtx>(this->ifft->transpose());

    ASSERT_EQ(transp->get_size(), this->ifft->get_size());
    ASSERT_TRUE(transp->is_inverse());
}


TYPED_TEST(Fft, IsConjTransposable1D)
{
    auto transp =
        gko::as<typename TestFixture::Mtx>(this->fft->conj_transpose());

    ASSERT_EQ(transp->get_size(), this->fft->get_size());
    ASSERT_TRUE(transp->is_inverse());
}


TYPED_TEST(Fft, InverseIsConjTransposable1D)
{
    auto transp =
        gko::as<typename TestFixture::Mtx>(this->ifft->conj_transpose());

    ASSERT_EQ(transp->get_size(), this->ifft->get_size());
    ASSERT_FALSE(transp->is_inverse());
}


TYPED_TEST(Fft, IsTransposable2D)
{
    auto transp = gko::as<typename TestFixture::Mtx2>(this->fft2->transpose());

    ASSERT_EQ(transp->get_size(), this->fft2->get_size());
    ASSERT_FALSE(transp->is_inverse());
}


TYPED_TEST(Fft, InverseIsTransposable2D)
{
    auto transp = gko::as<typename TestFixture::Mtx2>(this->ifft2->transpose());

    ASSERT_EQ(transp->get_size(), this->ifft2->get_size());
    ASSERT_TRUE(transp->is_inverse());
}


TYPED_TEST(Fft, IsConjTransposable2D)
{
    auto transp =
        gko::as<typename TestFixture::Mtx2>(this->fft2->conj_transpose());

    ASSERT_EQ(transp->get_size(), this->fft2->get_size());
    ASSERT_TRUE(transp->is_inverse());
}


TYPED_TEST(Fft, InverseIsConjTransposable2D)
{
    auto transp =
        gko::as<typename TestFixture::Mtx2>(this->ifft2->conj_transpose());

    ASSERT_EQ(transp->get_size(), this->ifft2->get_size());
    ASSERT_FALSE(transp->is_inverse());
}


TYPED_TEST(Fft, IsTransposable3D)
{
    auto transp = gko::as<typename TestFixture::Mtx3>(this->fft3->transpose());

    ASSERT_EQ(transp->get_size(), this->fft3->get_size());
    ASSERT_FALSE(transp->is_inverse());
}


TYPED_TEST(Fft, InverseIsTransposable3D)
{
    auto transp = gko::as<typename TestFixture::Mtx3>(this->ifft3->transpose());

    ASSERT_EQ(transp->get_size(), this->ifft3->get_size());
    ASSERT_TRUE(transp->is_inverse());
}


TYPED_TEST(Fft, IsConjTransposable3D)
{
    auto transp =
        gko::as<typename TestFixture::Mtx3>(this->fft3->conj_transpose());

    ASSERT_EQ(transp->get_size(), this->fft3->get_size());
    ASSERT_TRUE(transp->is_inverse());
}


TYPED_TEST(Fft, InverseIsConjTransposable3D)
{
    auto transp =
        gko::as<typename TestFixture::Mtx3>(this->ifft3->conj_transpose());

    ASSERT_EQ(transp->get_size(), this->ifft3->get_size());
    ASSERT_FALSE(transp->is_inverse());
}


TYPED_TEST(Fft, Applies1DToDense)
{
    using T = typename TestFixture::value_type;
    auto out = this->amplitude->clone();

    this->fft->apply(this->amplitude, out);

    GKO_ASSERT_MTX_NEAR(out, this->frequency1, r<T>::value);
}


TYPED_TEST(Fft, AppliesStrided1DToDense)
{
    using T = typename TestFixture::value_type;
    auto in_view =
        this->amplitude->create_submatrix({0, this->n}, {0, this->subcols});
    auto ref_view =
        this->frequency1->create_submatrix({0, this->n}, {0, this->subcols});
    auto out =
        TestFixture::Vec::create(this->exec, in_view->get_size(), this->stride);

    this->fft->apply(in_view, out);

    GKO_ASSERT_MTX_NEAR(out, ref_view, r<T>::value);
}


TYPED_TEST(Fft, AppliesInverse1DToDense)
{
    using T = typename TestFixture::value_type;
    auto out = this->frequency1->clone();

    this->ifft->apply(this->frequency1, out);
    out->scale(this->inv_n_scalar);

    GKO_ASSERT_MTX_NEAR(out, this->amplitude, r<T>::value);
}


TYPED_TEST(Fft, AppliesStridedInverse1DToDense)
{
    using T = typename TestFixture::value_type;
    auto in_view =
        this->frequency1->create_submatrix({0, this->n}, {0, this->subcols});
    auto ref_view =
        this->amplitude->create_submatrix({0, this->n}, {0, this->subcols});
    auto out =
        TestFixture::Vec::create(this->exec, in_view->get_size(), this->stride);

    this->ifft->apply(in_view, out);
    out->scale(this->inv_n_scalar);

    GKO_ASSERT_MTX_NEAR(out, ref_view, r<T>::value);
}


TYPED_TEST(Fft, Applies2DToDense)
{
    using T = typename TestFixture::value_type;
    auto out = this->amplitude->clone();

    this->fft2->apply(this->amplitude, out);

    GKO_ASSERT_MTX_NEAR(out, this->frequency2, r<T>::value);
}


TYPED_TEST(Fft, AppliesStrided2DToDense)
{
    using T = typename TestFixture::value_type;
    auto in_view =
        this->amplitude->create_submatrix({0, this->n}, {0, this->subcols});
    auto ref_view =
        this->frequency2->create_submatrix({0, this->n}, {0, this->subcols});
    auto out =
        TestFixture::Vec::create(this->exec, in_view->get_size(), this->stride);

    this->fft2->apply(in_view, out);

    GKO_ASSERT_MTX_NEAR(out, ref_view, r<T>::value);
}


TYPED_TEST(Fft, AppliesInverse2DToDense)
{
    using T = typename TestFixture::value_type;
    auto out = this->frequency2->clone();

    this->ifft2->apply(this->frequency2, out);
    out->scale(this->inv_n_scalar);

    GKO_ASSERT_MTX_NEAR(out, this->amplitude, r<T>::value);
}


TYPED_TEST(Fft, AppliesStridedInverse2DToDense)
{
    using T = typename TestFixture::value_type;
    auto in_view =
        this->frequency2->create_submatrix({0, this->n}, {0, this->subcols});
    auto ref_view =
        this->amplitude->create_submatrix({0, this->n}, {0, this->subcols});
    auto out =
        TestFixture::Vec::create(this->exec, in_view->get_size(), this->stride);

    this->ifft2->apply(in_view, out);
    out->scale(this->inv_n_scalar);

    GKO_ASSERT_MTX_NEAR(out, ref_view, r<T>::value);
}


TYPED_TEST(Fft, Applies3DToDense)
{
    using T = typename TestFixture::value_type;
    auto out = this->amplitude->clone();

    this->fft3->apply(this->amplitude, out);

    GKO_ASSERT_MTX_NEAR(out, this->frequency3, r<T>::value);
}


TYPED_TEST(Fft, AppliesStrided3DToDense)
{
    using T = typename TestFixture::value_type;
    auto in_view =
        this->amplitude->create_submatrix({0, this->n}, {0, this->subcols});
    auto ref_view =
        this->frequency3->create_submatrix({0, this->n}, {0, this->subcols});
    auto out =
        TestFixture::Vec::create(this->exec, in_view->get_size(), this->stride);

    this->fft3->apply(in_view, out);

    GKO_ASSERT_MTX_NEAR(out, ref_view, r<T>::value);
}


TYPED_TEST(Fft, AppliesInverse3DToDense)
{
    using T = typename TestFixture::value_type;
    auto out = this->frequency3->clone();

    this->ifft3->apply(this->frequency3, out);
    out->scale(this->inv_n_scalar);

    GKO_ASSERT_MTX_NEAR(out, this->amplitude, r<T>::value);
}


TYPED_TEST(Fft, AppliesStridedInverse3DToDense)
{
    using T = typename TestFixture::value_type;
    auto in_view =
        this->frequency3->create_submatrix({0, this->n}, {0, this->subcols});
    auto ref_view =
        this->amplitude->create_submatrix({0, this->n}, {0, this->subcols});
    auto out =
        TestFixture::Vec::create(this->exec, in_view->get_size(), this->stride);

    this->ifft3->apply(in_view, out);
    out->scale(this->inv_n_scalar);

    GKO_ASSERT_MTX_NEAR(out, ref_view, r<T>::value);
}


TYPED_TEST(Fft, Writes1DFFTToMatrixData32)
{
    using T = typename TestFixture::value_type;
    gko::matrix_data<T, gko::int32> data;
    auto output = gko::matrix::Dense<T>::create(this->exec);

    this->fft->write(data);
    output->read(data);

    GKO_ASSERT_MTX_NEAR(output, this->dense_fft, r<T>::value);
}


TYPED_TEST(Fft, Writes1DFFTToMatrixData64)
{
    using T = typename TestFixture::value_type;
    gko::matrix_data<T, gko::int64> data;
    auto output = gko::matrix::Dense<T>::create(this->exec);

    this->fft->write(data);
    output->read(data);

    GKO_ASSERT_MTX_NEAR(output, this->dense_fft, r<T>::value);
}


TYPED_TEST(Fft, Writes2DFFTToMatrixData32)
{
    using T = typename TestFixture::value_type;
    gko::matrix_data<T, gko::int32> data;
    auto output = gko::matrix::Dense<T>::create(this->exec);

    this->fft2->write(data);
    output->read(data);

    GKO_ASSERT_MTX_NEAR(output, this->dense_fft2, r<T>::value);
}


TYPED_TEST(Fft, Writes2DFFTToMatrixData64)
{
    using T = typename TestFixture::value_type;
    gko::matrix_data<T, gko::int64> data;
    auto output = gko::matrix::Dense<T>::create(this->exec);

    this->fft2->write(data);
    output->read(data);

    GKO_ASSERT_MTX_NEAR(output, this->dense_fft2, r<T>::value);
}


TYPED_TEST(Fft, Writes3DFFTToMatrixData32)
{
    using T = typename TestFixture::value_type;
    gko::matrix_data<T, gko::int32> data;
    auto output = gko::matrix::Dense<T>::create(this->exec);

    this->fft3->write(data);
    output->read(data);

    GKO_ASSERT_MTX_NEAR(output, this->dense_fft3, r<T>::value);
}


TYPED_TEST(Fft, Writes3DFFTToMatrixData64)
{
    using T = typename TestFixture::value_type;
    gko::matrix_data<T, gko::int64> data;
    auto output = gko::matrix::Dense<T>::create(this->exec);

    this->fft3->write(data);
    output->read(data);

    GKO_ASSERT_MTX_NEAR(output, this->dense_fft3, r<T>::value);
}


TYPED_TEST(Fft, Writes1DInvFFTToMatrixData32)
{
    using T = typename TestFixture::value_type;
    gko::matrix_data<T, gko::int32> data;
    auto output = gko::matrix::Dense<T>::create(this->exec);

    this->ifft->write(data);
    output->read(data);

    GKO_ASSERT_MTX_NEAR(output, this->dense_ifft, r<T>::value);
}


TYPED_TEST(Fft, Writes1DInvFFTToMatrixData64)
{
    using T = typename TestFixture::value_type;
    gko::matrix_data<T, gko::int64> data;
    auto output = gko::matrix::Dense<T>::create(this->exec);

    this->ifft->write(data);
    output->read(data);

    GKO_ASSERT_MTX_NEAR(output, this->dense_ifft, r<T>::value);
}


TYPED_TEST(Fft, Writes2DInvFFTToMatrixData32)
{
    using T = typename TestFixture::value_type;
    gko::matrix_data<T, gko::int32> data;
    auto output = gko::matrix::Dense<T>::create(this->exec);

    this->ifft2->write(data);
    output->read(data);

    GKO_ASSERT_MTX_NEAR(output, this->dense_ifft2, r<T>::value);
}


TYPED_TEST(Fft, Writes2DInvFFTToMatrixData64)
{
    using T = typename TestFixture::value_type;
    gko::matrix_data<T, gko::int64> data;
    auto output = gko::matrix::Dense<T>::create(this->exec);

    this->ifft2->write(data);
    output->read(data);

    GKO_ASSERT_MTX_NEAR(output, this->dense_ifft2, r<T>::value);
}


TYPED_TEST(Fft, Writes3DInvFFTToMatrixData32)
{
    using T = typename TestFixture::value_type;
    gko::matrix_data<T, gko::int32> data;
    auto output = gko::matrix::Dense<T>::create(this->exec);

    this->ifft3->write(data);
    output->read(data);

    GKO_ASSERT_MTX_NEAR(output, this->dense_ifft3, r<T>::value);
}


TYPED_TEST(Fft, Writes3DInvFFTToMatrixData64)
{
    using T = typename TestFixture::value_type;
    gko::matrix_data<T, gko::int64> data;
    auto output = gko::matrix::Dense<T>::create(this->exec);

    this->ifft3->write(data);
    output->read(data);

    GKO_ASSERT_MTX_NEAR(output, this->dense_ifft3, r<T>::value);
}


}  // namespace
