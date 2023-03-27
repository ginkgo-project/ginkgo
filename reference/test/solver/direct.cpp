/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <ginkgo/core/solver/direct.hpp>


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/factorization/lu.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/matrix_generator.hpp"
#include "matrices/config.hpp"


template <typename ValueIndexType>
class Direct : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using matrix_type = gko::matrix::Csr<value_type, index_type>;
    using vector_type = gko::matrix::Dense<value_type>;
    using solver_type =
        gko::experimental::solver::Direct<value_type, index_type>;
    Direct() : rng{93671}, exec(gko::ReferenceExecutor::create()) {}

    void setup(const char* mtx_filename, int nrhs = 1)
    {
        std::ifstream stream{mtx_filename};
        mtx = gko::read<matrix_type>(stream, exec);
        auto factory =
            solver_type::build()
                .with_factorization(
                    gko::experimental::factorization::Lu<value_type,
                                                         index_type>::build()
                        .with_symmetric_sparsity(true)
                        .on(exec))
                .on(exec);
        solver = factory->generate(mtx);
        std::normal_distribution<> dist(0, 1);
        x = gko::test::generate_random_dense_matrix<value_type>(
            mtx->get_size()[0], nrhs, dist, rng, this->exec);
        x_ref = x->clone();
        b = x->clone();
        mtx->apply(x, b);
    }

    std::default_random_engine rng;
    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<matrix_type> mtx;
    std::unique_ptr<vector_type> b;
    std::unique_ptr<vector_type> x;
    std::unique_ptr<vector_type> x_ref;
    std::unique_ptr<solver_type> solver;
};

TYPED_TEST_SUITE(Direct, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Direct, SolvesAni1SingleRhs)
{
    using matrix_type = typename TestFixture::matrix_type;
    using value_type = typename TestFixture::value_type;
    this->setup(gko::matrices::location_ani1_mtx);

    this->solver->apply(this->b, this->x);

    GKO_ASSERT_MTX_NEAR(this->x, this->x_ref, r<value_type>::value);
}


TYPED_TEST(Direct, SolvesAni1AmdMultipleRhs)
{
    using matrix_type = typename TestFixture::matrix_type;
    using value_type = typename TestFixture::value_type;
    this->setup(gko::matrices::location_ani1_amd_mtx, 3);

    this->solver->apply(this->b, this->x);

    GKO_ASSERT_MTX_NEAR(this->x, this->x_ref, r<value_type>::value);
}
