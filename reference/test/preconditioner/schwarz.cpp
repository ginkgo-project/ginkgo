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

#include <ginkgo/core/preconditioner/schwarz.hpp>


#include <algorithm>
#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/base/extended_float.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class Schwarz : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using CsrMat = gko::matrix::Csr<value_type, index_type>;
    using Prec = gko::preconditioner::Schwarz<value_type, index_type>;
    using InnerSolver = gko::preconditioner::Jacobi<value_type, index_type>;
    // using InnerSolver = gko::solver::Cg<value_type>;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using mdata = gko::matrix_data<value_type, index_type>;

    Schwarz()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::matrix::Csr<value_type, index_type>::create(
              exec, gko::dim<2>{5}, 13))
    {
        inner_factory =
            gko::share(InnerSolver::build().with_max_block_size(5u).on(exec));
        // inner_factory = gko::share(
        //     InnerSolver::build()
        //         .with_criteria(
        //             gko::stop::Iteration::build().with_max_iters(100u).on(exec),
        //             gko::stop::ResidualNorm<value_type>::build()
        //                 .with_reduction_factor(
        //                     gko::remove_complex<value_type>(1e-16))
        //                 .on(exec))
        //         .on(exec));
        schwarz_factory = gko::share(
            Prec::build()
                .with_subdomain_sizes(std::vector<gko::size_type>{2, 2, 1})
                .with_inner_solver(inner_factory)
                .on(exec));
        /* test matrix:
            4  -2 |       |-2
           -1   4 |       |
           -------+-------+--
                  | 4  -2 |
                  |-1   4 |-2
           -------+-------+--
           -1     |    -1 | 4
         */
        init_array<index_type>(mtx->get_row_ptrs(), {0, 3, 5, 7, 10, 13});
        init_array<index_type>(mtx->get_col_idxs(),
                               {0, 1, 4, 0, 1, 2, 3, 2, 3, 4, 0, 3, 4});
        init_array<value_type>(mtx->get_values(),
                               {4.0, -2.0, -2.0, -1.0, 4.0, 4.0, -2.0, -1.0,
                                4.0, -2.0, -1.0, -1.0, 4.0});
    }

    template <typename T>
    void init_array(T* arr, std::initializer_list<T> vals)
    {
        for (auto elem : vals) {
            *(arr++) = elem;
        }
    }

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<typename Prec::Factory> schwarz_factory;
    std::shared_ptr<typename InnerSolver::Factory> inner_factory;
    std::shared_ptr<CsrMat> mtx;
};

TYPED_TEST_SUITE(Schwarz, gko::test::ValueIndexTypes);


TYPED_TEST(Schwarz, CanBeGenerated)
{
    using Csr = typename TestFixture::CsrMat;
    using T = typename TestFixture::value_type;
    auto schwarz = this->schwarz_factory->generate(this->mtx);

    ASSERT_NE(schwarz, nullptr);
    EXPECT_EQ(schwarz->get_executor(), this->exec);
    EXPECT_EQ(schwarz->get_num_subdomains(), 3);
    ASSERT_EQ(schwarz->get_size(), gko::dim<2>(5, 5));
    auto mat1 =
        gko::initialize<Csr>({I<T>{4.0, -2.0}, I<T>{-1.0, 4.0}}, this->exec);
    auto mat2 =
        gko::initialize<Csr>({I<T>{4.0, -2.0}, I<T>{-1.0, 4.0}}, this->exec);
    auto mat3 = gko::initialize<Csr>({I<T>{4.0}}, this->exec);
    auto subd_mats = schwarz->get_subdomain_matrices();
    GKO_ASSERT_MTX_NEAR(mat1, static_cast<Csr*>(subd_mats[0].get()), 0.0);
    GKO_ASSERT_MTX_NEAR(mat2, static_cast<Csr*>(subd_mats[1].get()), 0.0);
    GKO_ASSERT_MTX_NEAR(mat3, static_cast<Csr*>(subd_mats[2].get()), 0.0);
}


TYPED_TEST(Schwarz, AppliesToVector)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    auto x = gko::initialize<Vec>({1.0, -1.0, 2.0, -2.0, 3.0}, this->exec);
    auto b = gko::initialize<Vec>({4.0, -1.0, -2.0, 4.0, -1.0}, this->exec);
    auto schwarz = this->schwarz_factory->generate(this->mtx);

    schwarz->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 0.0, 0.0, 1.0, -0.25}),
                        r<value_type>::value);
}


TYPED_TEST(Schwarz, AppliesToMultipleVectors)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto x =
        gko::initialize<Vec>(2,
                             {I<T>{1.0, 0.5}, I<T>{-1.0, -0.5}, I<T>{2.0, 1.0},
                              I<T>{-2.0, -1.0}, I<T>{3.0, 1.5}},
                             this->exec);
    auto b =
        gko::initialize<Vec>(2,
                             {I<T>{4.0, -2.0}, I<T>{-1.0, 4.0}, I<T>{-2.0, 4.0},
                              I<T>{4.0, -1.0}, I<T>{-1.0, 4.0}},
                             this->exec);
    auto schwarz = this->schwarz_factory->generate(this->mtx);

    schwarz->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(
        x, l({{1.0, 0.0}, {0.0, 1.0}, {0.0, 1.0}, {1.0, 0.0}, {-0.25, 1.0}}),
        r<value_type>::value);
}


TYPED_TEST(Schwarz, AppliesLinearCombinationToVector)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    auto x = gko::initialize<Vec>({1.0, -1.0, 2.0, -2.0, 3.0}, this->exec);
    auto b = gko::initialize<Vec>({4.0, -1.0, -2.0, 4.0, -1.0}, this->exec);
    auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto schwarz = this->schwarz_factory->generate(this->mtx);

    schwarz->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 1.0, -2.0, 4.0, -3.5}),
                        r<value_type>::value);
}


TYPED_TEST(Schwarz, AppliesLinearCombinationToMultipleVectors)
{
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto half_tol = std::sqrt(r<value_type>::value);
    auto x =
        gko::initialize<Vec>(2,
                             {I<T>{1.0, 0.5}, I<T>{-1.0, -0.5}, I<T>{2.0, 1.0},
                              I<T>{-2.0, -1.0}, I<T>{3.0, 1.5}},
                             this->exec);
    auto b =
        gko::initialize<Vec>(2,
                             {I<T>{4.0, -2.0}, I<T>{-1.0, 4.0}, I<T>{-2.0, 4.0},
                              I<T>{4.0, -1.0}, I<T>{-1.0, 4.0}},
                             this->exec);
    auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto schwarz = this->schwarz_factory->generate(this->mtx);

    schwarz->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(
        x, l({{1.0, -0.5}, {1.0, 2.5}, {-2.0, 1.0}, {4.0, 1.0}, {-3.5, 0.5}}),
        r<value_type>::value);
}


}  // namespace
