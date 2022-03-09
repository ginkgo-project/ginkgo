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

#include <ginkgo/core/matrix/csr.hpp>


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/test/utils/matrix_utils.hpp"
#include "core/test/utils/unsort_matrix.hpp"
#include "cuda/test/utils.hpp"


namespace {


class Csr : public ::testing::Test {
protected:
    using Arr = gko::Array<int>;
    using Vec = gko::matrix::Dense<>;
    using Mtx = gko::matrix::Csr<>;
    using ComplexVec = gko::matrix::Dense<std::complex<double>>;
    using ComplexMtx = gko::matrix::Csr<std::complex<double>>;

    Csr()
#ifdef GINKGO_FAST_TESTS
        : mtx_size(152, 231),
#else
        : mtx_size(532, 231),
#endif
          rand_engine(42)
    {}

    void SetUp()
    {
        ASSERT_GT(gko::CudaExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        cuda = gko::CudaExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (cuda != nullptr) {
            ASSERT_NO_THROW(cuda->synchronize());
        }
    }

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols,
                                     int min_nnz_row)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(min_nnz_row, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void set_up_apply_data()
    {
        mtx2 = Mtx::create(ref);
        mtx2->copy_from(gen_mtx<Mtx>(mtx_size[0], mtx_size[1], 5));
        dmtx2 = Mtx::create(cuda);
        dmtx2->copy_from(mtx2.get());
    }

    void set_up_apply_data(std::shared_ptr<Mtx::strategy_type> strategy,
                           int num_vectors = 1)
    {
        mtx = Mtx::create(ref, strategy);
        mtx->copy_from(gen_mtx<Vec>(mtx_size[0], mtx_size[1], 1));
        square_mtx = Mtx::create(ref, strategy);
        square_mtx->copy_from(gen_mtx<Vec>(mtx_size[0], mtx_size[0], 1));
        expected = gen_mtx<Vec>(mtx_size[0], num_vectors, 1);
        expected2 = gen_mtx<Vec>(mtx_size[0], num_vectors, 1);
        y = gen_mtx<Vec>(mtx_size[1], num_vectors, 1);
        y2 = gen_mtx<Vec>(mtx_size[0], num_vectors, 1);
        alpha = gko::initialize<Vec>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        dmtx = Mtx::create(cuda, strategy);
        dmtx->copy_from(mtx.get());
        square_dmtx = Mtx::create(cuda, strategy);
        square_dmtx->copy_from(square_mtx.get());
        dresult = gko::clone(cuda, expected);
        dresult2 = gko::clone(cuda, expected2);
        dy = gko::clone(cuda, y);
        dy2 = gko::clone(cuda, y2);
        dalpha = gko::clone(cuda, alpha);
        dbeta = gko::clone(cuda, beta);

        std::vector<int> tmp(mtx->get_size()[0], 0);
        auto rng = std::default_random_engine{};
        std::iota(tmp.begin(), tmp.end(), 0);
        std::shuffle(tmp.begin(), tmp.end(), rng);
        std::vector<int> tmp2(mtx->get_size()[1], 0);
        std::iota(tmp2.begin(), tmp2.end(), 0);
        std::shuffle(tmp2.begin(), tmp2.end(), rng);
        rpermute_idxs = std::make_unique<Arr>(ref, tmp.begin(), tmp.end());
        cpermute_idxs = std::make_unique<Arr>(ref, tmp2.begin(), tmp2.end());
    }

    void set_up_apply_complex_data(
        std::shared_ptr<ComplexMtx::strategy_type> strategy)
    {
        complex_mtx = ComplexMtx::create(ref, strategy);
        complex_mtx->copy_from(
            gen_mtx<ComplexVec>(mtx_size[0], mtx_size[1], 1));
        complex_dmtx = ComplexMtx::create(cuda, strategy);
        complex_dmtx->copy_from(complex_mtx.get());
    }

    void unsort_mtx()
    {
        gko::test::unsort_matrix(mtx.get(), rand_engine);
        dmtx->copy_from(mtx.get());
    }


    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::CudaExecutor> cuda;

    const gko::dim<2> mtx_size;
    std::default_random_engine rand_engine;

    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Mtx> mtx2;
    std::unique_ptr<ComplexMtx> complex_mtx;
    std::unique_ptr<Mtx> square_mtx;
    std::unique_ptr<Vec> expected;
    std::unique_ptr<Vec> y;
    std::unique_ptr<Vec> expected2;
    std::unique_ptr<Vec> y2;
    std::unique_ptr<Vec> alpha;
    std::unique_ptr<Vec> beta;

    std::unique_ptr<Mtx> dmtx;
    std::unique_ptr<Mtx> dmtx2;
    std::unique_ptr<ComplexMtx> complex_dmtx;
    std::unique_ptr<Mtx> square_dmtx;
    std::unique_ptr<Vec> dresult;
    std::unique_ptr<Vec> dy;
    std::unique_ptr<Vec> dresult2;
    std::unique_ptr<Vec> dy2;
    std::unique_ptr<Vec> dalpha;
    std::unique_ptr<Vec> dbeta;
    std::unique_ptr<Arr> rpermute_idxs;
    std::unique_ptr<Arr> cpermute_idxs;
};


TEST_F(Csr, AsyncSimpleApplyIsEquivalentToRefWithLoadBalanceUnsorted)
{
    set_up_apply_data(std::make_shared<Mtx::load_balance>(cuda));
    unsort_mtx();

    dmtx->apply(dy.get(), dresult.get(), cuda->get_default_exec_stream())
        ->wait();
    auto hand2 =
        square_dmtx->apply(dy2.get(), dresult2.get(), cuda->get_handle_at(1));
    mtx->apply(y.get(), expected.get());
    square_mtx->apply(y2.get(), expected2.get());
    hand2->wait();

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-14);
}


TEST_F(Csr, AsyncSimpleApplyIsEquivalentToRefWithLoadBalanceUnsorted2)
{
    set_up_apply_data(std::make_shared<Mtx::load_balance>(cuda));
    unsort_mtx();

    auto hand = dmtx->apply(dy.get(), dresult.get(), cuda->get_handle_at(0));
    auto hand2 =
        square_dmtx->apply(dy2.get(), dresult2.get(), cuda->get_handle_at(1));
    mtx->apply(y.get(), expected.get());
    square_mtx->apply(y2.get(), expected2.get());
    hand->wait();
    hand2->wait();

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
    GKO_ASSERT_MTX_NEAR(dresult2, expected2, 1e-14);
}


}  // namespace
