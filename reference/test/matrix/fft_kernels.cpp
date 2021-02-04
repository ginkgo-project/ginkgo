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

    Fft()
        : exec(gko::ReferenceExecutor::create()),
          n1{4},
          n2{8},
          n3{16},
          n{n1 * n2 * n3},
          subcols{3},
          stride{8},
          inv_n_scalar(gko::initialize<Vec>({T{1.f / n}}, exec)),
          fft(Mtx::create(exec, n)),
          fft2(Mtx2::create(exec, n1 * n2, n3)),
          fft3(Mtx3::create(exec, n1, n2, n3)),
          ifft(Mtx::create(exec, n, true)),
          ifft2(Mtx2::create(exec, n1 * n2, n3, true)),
          ifft3(Mtx3::create(exec, n1, n2, n3, true))
    {}

    std::unique_ptr<Vec> load(std::string filename)
    {
        auto input_file = std::ifstream(filename);
        if (!input_file) {
            throw gko::Error(__FILE__, __LINE__,
                             "Could not find the file \"" + filename +
                                 "\", which is required for this test.");
        }
        return gko::read<Vec>(input_file, exec);
    }

    void SetUp() override
    {
        amplitude = load(gko::matrices::location_fourier_in_mtx);
        frequency1 = load(gko::matrices::location_fourier_out1_mtx);
        frequency2 = load(gko::matrices::location_fourier_out2_mtx);
        frequency3 = load(gko::matrices::location_fourier_out3_mtx);
    }

    std::shared_ptr<const gko::Executor> exec;
    size_t n1;
    size_t n2;
    size_t n3;
    size_t n;
    size_t subcols;
    size_t stride;
    std::unique_ptr<Vec> inv_n_scalar;
    std::unique_ptr<Vec> amplitude;
    std::unique_ptr<Vec> frequency1;
    std::unique_ptr<Vec> frequency2;
    std::unique_ptr<Vec> frequency3;
    std::unique_ptr<Mtx> fft;
    std::unique_ptr<Mtx2> fft2;
    std::unique_ptr<Mtx3> fft3;
    std::unique_ptr<Mtx> ifft;
    std::unique_ptr<Mtx2> ifft2;
    std::unique_ptr<Mtx3> ifft3;
};


TYPED_TEST_SUITE(Fft, gko::test::ComplexValueTypes);


TYPED_TEST(Fft, ThrowsOnNonPowerOfTwo1D)
{
    ASSERT_THROW(TestFixture::Mtx::create(this->exec, 3), gko::BadDimension);
}


TYPED_TEST(Fft, ThrowsOnNonPowerOfTwo2D)
{
    ASSERT_THROW(TestFixture::Mtx2::create(this->exec, 3, 5),
                 gko::BadDimension);
}


TYPED_TEST(Fft, ThrowsOnNonPowerOfTwo3D)
{
    ASSERT_THROW(TestFixture::Mtx3::create(this->exec, 3, 5, 7),
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

    this->fft->apply(this->amplitude.get(), out.get());

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

    this->fft->apply(in_view.get(), out.get());

    GKO_ASSERT_MTX_NEAR(out, ref_view, r<T>::value);
}


TYPED_TEST(Fft, AppliesInverse1DToDense)
{
    using T = typename TestFixture::value_type;
    auto out = this->frequency1->clone();

    this->ifft->apply(this->frequency1.get(), out.get());
    out->scale(this->inv_n_scalar.get());

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

    this->ifft->apply(in_view.get(), out.get());
    out->scale(this->inv_n_scalar.get());

    GKO_ASSERT_MTX_NEAR(out, ref_view, r<T>::value);
}


TYPED_TEST(Fft, Applies2DToDense)
{
    using T = typename TestFixture::value_type;
    auto out = this->amplitude->clone();

    this->fft2->apply(this->amplitude.get(), out.get());

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

    this->fft2->apply(in_view.get(), out.get());

    GKO_ASSERT_MTX_NEAR(out, ref_view, r<T>::value);
}


TYPED_TEST(Fft, AppliesInverse2DToDense)
{
    using T = typename TestFixture::value_type;
    auto out = this->frequency2->clone();

    this->ifft2->apply(this->frequency2.get(), out.get());
    out->scale(this->inv_n_scalar.get());

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

    this->ifft2->apply(in_view.get(), out.get());
    out->scale(this->inv_n_scalar.get());

    GKO_ASSERT_MTX_NEAR(out, ref_view, r<T>::value);
}


TYPED_TEST(Fft, Applies3DToDense)
{
    using T = typename TestFixture::value_type;
    auto out = this->amplitude->clone();

    this->fft3->apply(this->amplitude.get(), out.get());

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

    this->fft3->apply(in_view.get(), out.get());

    GKO_ASSERT_MTX_NEAR(out, ref_view, r<T>::value);
}


TYPED_TEST(Fft, AppliesInverse3DToDense)
{
    using T = typename TestFixture::value_type;
    auto out = this->frequency3->clone();

    this->ifft3->apply(this->frequency3.get(), out.get());
    out->scale(this->inv_n_scalar.get());

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

    this->ifft3->apply(in_view.get(), out.get());
    out->scale(this->inv_n_scalar.get());

    GKO_ASSERT_MTX_NEAR(out, ref_view, r<T>::value);
}


}  // namespace
