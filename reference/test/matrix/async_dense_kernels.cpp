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


#include <complex>
#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/matrix/dense_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename T>
class Dense : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using MixedMtx = gko::matrix::Dense<gko::next_precision<value_type>>;
    using ComplexMtx = gko::to_complex<Mtx>;
    using MixedComplexMtx = gko::to_complex<MixedMtx>;
    using RealMtx = gko::remove_complex<Mtx>;
    Dense()
        : exec(gko::ReferenceExecutor::create(2)),
          mtx1(gko::initialize<Mtx>(4, {{1.0, 2.0, 3.0}, {1.5, 2.5, 3.5}},
                                    exec)),
          mtx2(gko::initialize<Mtx>({I<T>({1.0, -1.0}), I<T>({-2.0, 2.0})},
                                    exec)),
          mtx3(gko::initialize<Mtx>(4, {{1.0, 2.0, 3.0}, {0.5, 1.5, 2.5}},
                                    exec)),
          mtx4(gko::initialize<Mtx>(4, {{1.0, 3.0, 2.0}, {0.0, 5.0, 0.0}},
                                    exec)),
          mtx5(gko::initialize<Mtx>(
              {{1.0, -1.0, -0.5}, {-2.0, 2.0, 4.5}, {2.1, 3.4, 1.2}}, exec)),
          mtx6(gko::initialize<Mtx>({{1.0, 2.0, 0.0}, {0.0, 1.5, 0.0}}, exec)),
          mtx7(gko::initialize<Mtx>({{1.0, 2.0, 3.0}, {0.0, 1.5, 0.0}}, exec)),
          mtx8(gko::initialize<Mtx>(
              {I<T>({1.0, -1.0}), I<T>({-2.0, 2.0}), I<T>({-3.0, 3.0})}, exec)),
          mtx9(gko::initialize<Mtx>({{1.0, 2.0, 2.0, 8.0, 3.0},
                                     {0.0, 3.0, 1.5, 2.0, 0.0},
                                     {0.0, 3.0, 2.5, 1.5, 0.0},
                                     {1.0, 2.0, 1.0, 3.0, 4.0},
                                     {1.0, 1.0, 2.0, 1.5, 3.0}},
                                    exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx1;
    std::unique_ptr<Mtx> mtx2;
    std::unique_ptr<Mtx> mtx3;
    std::unique_ptr<Mtx> mtx4;
    std::unique_ptr<Mtx> mtx5;
    std::unique_ptr<Mtx> mtx6;
    std::unique_ptr<Mtx> mtx7;
    std::unique_ptr<Mtx> mtx8;
    std::unique_ptr<Mtx> mtx9;

    std::default_random_engine rand_engine;

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<gko::size_type>(num_cols, num_cols),
            std::normal_distribution<gko::remove_complex<value_type>>(0.0, 1.0),
            rand_engine, exec);
    }
};


TYPED_TEST_SUITE(Dense, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Dense, AsyncAppliesToDense)
{
    using T = typename TestFixture::value_type;
    T in_stride{-1};
    this->mtx3->get_values()[3] = in_stride;

    auto hand = this->mtx2->apply(this->mtx1.get(), this->mtx3.get(),
                                  this->exec->get_handle_at(0));

    EXPECT_NE(this->mtx3->at(0, 0), T{-0.5});
    hand->wait();
    EXPECT_EQ(this->mtx3->at(0, 0), T{-0.5});
    EXPECT_EQ(this->mtx3->at(0, 1), T{-0.5});
    EXPECT_EQ(this->mtx3->at(0, 2), T{-0.5});
    EXPECT_EQ(this->mtx3->at(1, 0), T{1.0});
    EXPECT_EQ(this->mtx3->at(1, 1), T{1.0});
    EXPECT_EQ(this->mtx3->at(1, 2), T{1.0});
    ASSERT_EQ(this->mtx3->get_values()[3], in_stride);
}


// FIXME: Some segfault with mixed precision applies. Probably some runaway
// thread
// TYPED_TEST(Dense, AsyncAppliesToMixedDense)
// {
//     using MixedMtx = typename TestFixture::MixedMtx;
//     using MixedT = typename MixedMtx::value_type;
//     auto mmtx1 = MixedMtx::create(this->exec);
//     auto mmtx3 = MixedMtx::create(this->exec);
//     this->mtx1->convert_to(mmtx1.get());
//     this->mtx3->convert_to(mmtx3.get());

//     auto hand = this->mtx2->apply(mmtx1.get(), mmtx3.get(),
//                                   this->exec->get_handle_at(0));

//     EXPECT_NE(mmtx3->at(0, 0), MixedT{-0.5});
//     hand->wait();
//     EXPECT_EQ(mmtx3->at(0, 0), MixedT{-0.5});
//     EXPECT_EQ(mmtx3->at(0, 1), MixedT{-0.5});
//     EXPECT_EQ(mmtx3->at(0, 2), MixedT{-0.5});
//     EXPECT_EQ(mmtx3->at(1, 0), MixedT{1.0});
//     EXPECT_EQ(mmtx3->at(1, 1), MixedT{1.0});
//     ASSERT_EQ(mmtx3->at(1, 2), MixedT{1.0});
// }


}  // namespace
