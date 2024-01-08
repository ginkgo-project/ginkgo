// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/fft.hpp>


#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


template <typename ValueType>
class Fft : public CommonTestFixture {
protected:
    using value_type = ValueType;
    using Vec = gko::matrix::Dense<value_type>;
    using Mtx = gko::matrix::Fft;
    using Mtx2 = gko::matrix::Fft2;
    using Mtx3 = gko::matrix::Fft3;

    Fft()
        : rand_engine(1364245),
          n1{16},
          n2{32},
          n3{64},
          n{n1 * n2 * n3},
          cols{3},
          subcols{2},
          out_stride{6}
    {
        data = gko::test::generate_random_matrix<Vec>(
            n, cols, std::uniform_int_distribution<>(1, cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
        ddata = Vec::create(exec);
        ddata->copy_from(this->data);
        data_strided = data->create_submatrix({0, n}, {0, subcols});
        ddata_strided = ddata->create_submatrix({0, n}, {0, subcols});
        out = data->clone();
        dout = data->clone();
        out_strided = Vec::create(ref, data_strided->get_size(), out_stride);
        dout_strided = Vec::create(exec, data_strided->get_size(), out_stride);
        fft = Mtx::create(ref, n);
        dfft = Mtx::create(exec, n);
        ifft = Mtx::create(ref, n, true);
        difft = Mtx::create(exec, n, true);
        fft2 = Mtx2::create(ref, n1 * n2, n3);
        dfft2 = Mtx2::create(exec, n1 * n2, n3);
        ifft2 = Mtx2::create(ref, n1 * n2, n3, true);
        difft2 = Mtx2::create(exec, n1 * n2, n3, true);
        fft3 = Mtx3::create(ref, n1, n2, n3);
        dfft3 = Mtx3::create(exec, n1, n2, n3);
        ifft3 = Mtx3::create(ref, n1, n2, n3, true);
        difft3 = Mtx3::create(exec, n1, n2, n3, true);
    }

    std::default_random_engine rand_engine;
    size_t n1;
    size_t n2;
    size_t n3;
    size_t n;
    size_t cols;
    size_t subcols;
    size_t out_stride;
    std::unique_ptr<Vec> data;
    std::unique_ptr<Vec> ddata;
    std::unique_ptr<Vec> data_strided;
    std::unique_ptr<Vec> ddata_strided;
    std::unique_ptr<Vec> out;
    std::unique_ptr<Vec> dout;
    std::unique_ptr<Vec> out_strided;
    std::unique_ptr<Vec> dout_strided;
    std::unique_ptr<Mtx> fft;
    std::unique_ptr<Mtx> dfft;
    std::unique_ptr<Mtx> ifft;
    std::unique_ptr<Mtx> difft;
    std::unique_ptr<Mtx2> fft2;
    std::unique_ptr<Mtx2> dfft2;
    std::unique_ptr<Mtx2> ifft2;
    std::unique_ptr<Mtx2> difft2;
    std::unique_ptr<Mtx3> fft3;
    std::unique_ptr<Mtx3> dfft3;
    std::unique_ptr<Mtx3> ifft3;
    std::unique_ptr<Mtx3> difft3;
};


TYPED_TEST_SUITE(Fft, gko::test::ComplexValueTypes, TypenameNameGenerator);


TYPED_TEST(Fft, Apply1DIsEqualToReference)
{
    using T = typename TestFixture::value_type;

    this->fft->apply(this->data, this->out);
    this->dfft->apply(this->ddata, this->dout);

    GKO_ASSERT_MTX_NEAR(this->out, this->dout, r<T>::value);
}


TYPED_TEST(Fft, ApplyStrided1DIsEqualToReference)
{
#if defined(GKO_COMPILING_HIP) && GINKGO_HIP_PLATFORM_HCC
    GTEST_SKIP() << "rocFFT has a bug related to strided 1D FFT";
#endif
    using T = typename TestFixture::value_type;

    this->fft->apply(this->data_strided, this->out_strided);
    this->dfft->apply(this->ddata_strided, this->dout_strided);

    GKO_ASSERT_MTX_NEAR(this->out_strided, this->dout_strided, r<T>::value);
}


TYPED_TEST(Fft, Apply1DInverseIsEqualToReference)
{
    using T = typename TestFixture::value_type;

    this->ifft->apply(this->data, this->out);
    this->difft->apply(this->ddata, this->dout);

    GKO_ASSERT_MTX_NEAR(this->out, this->dout, r<T>::value);
}


TYPED_TEST(Fft, ApplyStrided1DInverseIsEqualToReference)
{
#if defined(GKO_COMPILING_HIP) && GINKGO_HIP_PLATFORM_HCC
    GTEST_SKIP() << "rocFFT has a bug related to strided 1D FFT";
#endif
    using T = typename TestFixture::value_type;

    this->ifft->apply(this->data_strided, this->out_strided);
    this->difft->apply(this->ddata_strided, this->dout_strided);

    GKO_ASSERT_MTX_NEAR(this->out_strided, this->dout_strided, r<T>::value);
}


TYPED_TEST(Fft, Apply2DIsEqualToReference)
{
    using T = typename TestFixture::value_type;

    this->fft2->apply(this->data, this->out);
    this->dfft2->apply(this->ddata, this->dout);

    GKO_ASSERT_MTX_NEAR(this->out, this->dout, r<T>::value);
}


TYPED_TEST(Fft, ApplyStrided2DIsEqualToReference)
{
    using T = typename TestFixture::value_type;

    this->fft2->apply(this->data_strided, this->out_strided);
    this->dfft2->apply(this->ddata_strided, this->dout_strided);

    GKO_ASSERT_MTX_NEAR(this->out_strided, this->dout_strided, r<T>::value);
}


TYPED_TEST(Fft, Apply2DInverseIsEqualToReference)
{
    using T = typename TestFixture::value_type;

    this->ifft2->apply(this->data, this->out);
    this->difft2->apply(this->ddata, this->dout);

    GKO_ASSERT_MTX_NEAR(this->out, this->dout, r<T>::value);
}


TYPED_TEST(Fft, ApplyStrided2DInverseIsEqualToReference)
{
    using T = typename TestFixture::value_type;

    this->ifft2->apply(this->data_strided, this->out_strided);
    this->difft2->apply(this->ddata_strided, this->dout_strided);

    GKO_ASSERT_MTX_NEAR(this->out_strided, this->dout_strided, r<T>::value);
}


TYPED_TEST(Fft, Apply3DIsEqualToReference)
{
    using T = typename TestFixture::value_type;

    this->fft3->apply(this->data, this->out);
    this->dfft3->apply(this->ddata, this->dout);

    GKO_ASSERT_MTX_NEAR(this->out, this->dout, r<T>::value);
}


TYPED_TEST(Fft, ApplyStrided3DIsEqualToReference)
{
    using T = typename TestFixture::value_type;

    this->fft3->apply(this->data_strided, this->out_strided);
    this->dfft3->apply(this->ddata_strided, this->dout_strided);

    GKO_ASSERT_MTX_NEAR(this->out_strided, this->dout_strided, r<T>::value);
}


TYPED_TEST(Fft, Apply3DInverseIsEqualToReference)
{
    using T = typename TestFixture::value_type;

    this->ifft3->apply(this->data, this->out);
    this->difft3->apply(this->ddata, this->dout);

    GKO_ASSERT_MTX_NEAR(this->out, this->dout, r<T>::value);
}


TYPED_TEST(Fft, ApplyStrided3DInverseIsEqualToReference)
{
    using T = typename TestFixture::value_type;

    this->ifft3->apply(this->data_strided, this->out_strided);
    this->difft3->apply(this->ddata_strided, this->dout_strided);

    GKO_ASSERT_MTX_NEAR(this->out_strided, this->dout_strided, r<T>::value);
}
