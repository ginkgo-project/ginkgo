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

#include <ginkgo/core/matrix/dense.hpp>


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "hip/test/utils.hip.hpp"


namespace {


class Dense : public ::testing::Test {
protected:
    using itype = int;
    using vtype = double;
    using Mtx = gko::matrix::Dense<vtype>;
    using MixedMtx = gko::matrix::Dense<gko::next_precision<vtype>>;
    using NormVector = gko::matrix::Dense<gko::remove_complex<vtype>>;
    using Arr = gko::Array<itype>;
    using ComplexMtx = gko::matrix::Dense<std::complex<vtype>>;
    using Diagonal = gko::matrix::Diagonal<vtype>;
    using MixedComplexMtx =
        gko::matrix::Dense<gko::next_precision<std::complex<vtype>>>;

    Dense() : rand_engine(15) {}

    void SetUp()
    {
        ASSERT_GT(gko::HipExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        hip = gko::HipExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (hip != nullptr) {
            ASSERT_NO_THROW(hip->synchronize());
        }
    }

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(0.0, 1.0), rand_engine, ref);
    }

    void set_up_vector_data(gko::size_type num_vecs,
                            bool different_alpha = false)
    {
        x = gen_mtx<Mtx>(1000, num_vecs);
        y = gen_mtx<Mtx>(1000, num_vecs);
        if (different_alpha) {
            alpha = gen_mtx<Mtx>(1, num_vecs);
        } else {
            alpha = gko::initialize<Mtx>({2.0}, ref);
        }
        dx = gko::clone(hip, x);
        dy = gko::clone(hip, y);
        dalpha = gko::clone(hip, alpha);
        expected = Mtx::create(ref, gko::dim<2>{1, num_vecs});
        dresult = Mtx::create(hip, gko::dim<2>{1, num_vecs});
    }

    void set_up_apply_data()
    {
        x = gen_mtx<Mtx>(65, 25);
        c_x = gen_mtx<ComplexMtx>(65, 25);
        y = gen_mtx<Mtx>(25, 35);
        expected = gen_mtx<Mtx>(65, 35);
        alpha = gko::initialize<Mtx>({2.0}, ref);
        beta = gko::initialize<Mtx>({-1.0}, ref);
        square = gen_mtx<Mtx>(x->get_size()[0], x->get_size()[0]);
        dx = gko::clone(hip, x);
        dc_x = gko::clone(hip, c_x);
        dy = gko::clone(hip, y);
        dresult = gko::clone(hip, expected);
        dalpha = gko::clone(hip, alpha);
        dbeta = gko::clone(hip, beta);
        dsquare = gko::clone(hip, square);

        std::vector<itype> tmp(x->get_size()[0], 0);
        auto rng = std::default_random_engine{};
        std::iota(tmp.begin(), tmp.end(), 0);
        std::shuffle(tmp.begin(), tmp.end(), rng);
        std::vector<itype> tmp2(x->get_size()[1], 0);
        std::iota(tmp2.begin(), tmp2.end(), 0);
        std::shuffle(tmp2.begin(), tmp2.end(), rng);
        std::vector<itype> tmp3(x->get_size()[0] / 10);
        std::uniform_int_distribution<itype> row_dist(0, x->get_size()[0] - 1);
        for (auto& i : tmp3) {
            i = row_dist(rng);
        }
        rpermute_idxs =
            std::unique_ptr<Arr>(new Arr{ref, tmp.begin(), tmp.end()});
        cpermute_idxs =
            std::unique_ptr<Arr>(new Arr{ref, tmp2.begin(), tmp2.end()});
        rgather_idxs =
            std::unique_ptr<Arr>(new Arr{ref, tmp3.begin(), tmp3.end()});
    }

    template <typename ConvertedType, typename InputType>
    std::unique_ptr<ConvertedType> convert(InputType&& input)
    {
        auto result = ConvertedType::create(input->get_executor());
        input->convert_to(result.get());
        return result;
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::HipExecutor> hip;

    std::default_random_engine rand_engine;

    std::unique_ptr<Mtx> x;
    std::unique_ptr<ComplexMtx> c_x;
    std::unique_ptr<Mtx> y;
    std::unique_ptr<Mtx> alpha;
    std::unique_ptr<Mtx> beta;
    std::unique_ptr<Mtx> expected;
    std::unique_ptr<Mtx> square;
    std::unique_ptr<Mtx> dresult;
    std::unique_ptr<Mtx> dx;
    std::unique_ptr<ComplexMtx> dc_x;
    std::unique_ptr<Mtx> dy;
    std::unique_ptr<Mtx> dalpha;
    std::unique_ptr<Mtx> dbeta;
    std::unique_ptr<Mtx> dsquare;
    std::unique_ptr<Arr> rpermute_idxs;
    std::unique_ptr<Arr> cpermute_idxs;
    std::unique_ptr<Arr> rgather_idxs;
};


TEST_F(Dense, SimpleApplyIsEquivalentToRef)
{
    set_up_apply_data();

    auto hand2 =
        dx->apply(dy.get(), dresult.get(), this->hip->get_handle_at(0));
    x->apply(y.get(), expected.get());

    hand2->wait();

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Dense, SimpleApplyMixedIsEquivalentToRef)
{
    set_up_apply_data();

    auto hand2 =
        dx->apply(convert<MixedMtx>(dy).get(), convert<MixedMtx>(dresult).get(),
                  this->hip->get_handle_at(0));
    x->apply(convert<MixedMtx>(y).get(), convert<MixedMtx>(expected).get());

    hand2->wait();

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-7);
}


TEST_F(Dense, ApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data();
    auto complex_b = gen_mtx<ComplexMtx>(25, 1);
    auto dcomplex_b = gko::clone(hip, complex_b);
    auto complex_x = gen_mtx<ComplexMtx>(65, 1);
    auto dcomplex_x = gko::clone(hip, complex_x);

    auto hand2 = dx->apply(dcomplex_b.get(), dcomplex_x.get(),
                           this->hip->get_handle_at(0));
    x->apply(complex_b.get(), complex_x.get());

    hand2->wait();

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}


TEST_F(Dense, ApplyToMixedComplexIsEquivalentToRef)
{
    set_up_apply_data();
    auto complex_b = gen_mtx<MixedComplexMtx>(25, 1);
    auto dcomplex_b = gko::clone(hip, complex_b);
    auto complex_x = gen_mtx<MixedComplexMtx>(65, 1);
    auto dcomplex_x = gko::clone(hip, complex_x);

    auto hand2 = dx->apply(dcomplex_b.get(), dcomplex_x.get(),
                           this->hip->get_handle_at(0));
    x->apply(complex_b.get(), complex_x.get());

    hand2->wait();

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-7);
}


}  // namespace
