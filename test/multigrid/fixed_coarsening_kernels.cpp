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

#include <algorithm>
#include <fstream>
#include <random>
#include <string>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/row_gatherer.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>
#include <ginkgo/core/multigrid/fixed_coarsening.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/matrix_generator.hpp"
#include "core/test/utils/unsort_matrix.hpp"
#include "core/utils/matrix_utils.hpp"
#include "test/utils/executor.hpp"


namespace {


class FixedCoarsening : public ::testing::Test {
protected:
#if GINKGO_COMMON_SINGLE_MODE
    using value_type = float;
#else
    using value_type = double;
#endif  // GINKGO_COMMON_SINGLE_MODE
    using index_type = gko::int32;
    using Mtx = gko::matrix::Dense<value_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;

    FixedCoarsening() : rand_engine(30) {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        init_executor(ref, exec);
        m = 597;
    }

    void TearDown()
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    gko::Array<index_type> gen_coarse_array(gko::size_type num,
                                            gko::size_type num_rows)
    {
        GKO_ASSERT(num <= num_rows);
        gko::Array<index_type> coarse_array(ref, num);
        std::vector<index_type> base_vec(num_rows, 0);
        std::iota(base_vec.begin(), base_vec.end(), 0);
        std::shuffle(base_vec.begin(), base_vec.end(),
                     std::mt19937{std::random_device{}()});
        for (gko::size_type i = 0; i < num; i++) {
            coarse_array.get_data()[i] = base_vec[i + num_rows - num];
        }
        return coarse_array;
    }

    std::unique_ptr<Mtx> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<Mtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<value_type>(-1.0, 1.0), rand_engine, ref);
    }

    void initialize_data(gko::size_type coarse_size = 2)
    {
        coarse_rows = gen_coarse_array(coarse_size, m);
        c_dim = coarse_size;

        d_coarse_rows = gko::Array<index_type>(exec);
        d_coarse_rows = coarse_rows;
        restrict_op = Csr::create(ref, gko::dim<2>(c_dim, m), c_dim);

        d_restrict_op = Csr::create(exec);
        d_restrict_op->copy_from(restrict_op.get());
        auto system_dense = gen_mtx(m, m);
        system_mtx = Csr::create(ref);
        system_dense->convert_to(system_mtx.get());

        d_system_mtx = gko::clone(exec, system_mtx);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;

    std::default_random_engine rand_engine;

    gko::Array<index_type> coarse_rows;
    gko::Array<index_type> d_coarse_rows;

    std::shared_ptr<Csr> restrict_op;
    std::shared_ptr<Csr> d_restrict_op;

    std::shared_ptr<Csr> system_mtx;
    std::shared_ptr<Csr> d_system_mtx;

    gko::size_type n;
    gko::size_type m;
    gko::size_type c_dim;
};


TEST_F(FixedCoarsening, GenerateMgLevelIsEquivalentToRef)
{
    gko::size_type c_size = 34;
    initialize_data(c_size);
    auto mg_level_factory =
        gko::multigrid::FixedCoarsening<value_type, int>::build()
            .with_coarse_rows(coarse_rows)
            .with_skip_sorting(true)
            .on(ref);
    auto d_mg_level_factory =
        gko::multigrid::FixedCoarsening<value_type, int>::build()
            .with_coarse_rows(d_coarse_rows)
            .with_skip_sorting(true)
            .on(exec);

    auto mg_level = mg_level_factory->generate(system_mtx);
    auto d_mg_level = d_mg_level_factory->generate(d_system_mtx);

    GKO_ASSERT_MTX_NEAR(gko::as<Csr>(d_mg_level->get_prolong_op()),
                        gko::as<Csr>(mg_level->get_prolong_op()), 0.0);
    GKO_ASSERT_MTX_NEAR(gko::as<Csr>(d_mg_level->get_restrict_op()),
                        gko::as<Csr>(mg_level->get_restrict_op()), 0.0);
    GKO_ASSERT_MTX_NEAR(gko::as<Csr>(d_mg_level->get_coarse_op()),
                        gko::as<Csr>(mg_level->get_coarse_op()),
                        r<value_type>::value);
}


TEST_F(FixedCoarsening, GenerateMgLevelIsEquivalentToRefOnUnsortedMatrix)
{
    initialize_data(243);
    gko::test::unsort_matrix(gko::lend(system_mtx), rand_engine);
    d_system_mtx = gko::clone(exec, system_mtx);
    auto mg_level_factory =
        gko::multigrid::FixedCoarsening<value_type, int>::build()
            .with_coarse_rows(coarse_rows)
            .on(ref);
    auto d_mg_level_factory =
        gko::multigrid::FixedCoarsening<value_type, int>::build()
            .with_coarse_rows(d_coarse_rows)
            .on(exec);

    auto mg_level = mg_level_factory->generate(system_mtx);
    auto d_mg_level = d_mg_level_factory->generate(d_system_mtx);

    GKO_ASSERT_MTX_NEAR(gko::as<Csr>(d_mg_level->get_prolong_op()),
                        gko::as<Csr>(mg_level->get_prolong_op()), 0.0);
    GKO_ASSERT_MTX_NEAR(gko::as<Csr>(d_mg_level->get_restrict_op()),
                        gko::as<Csr>(mg_level->get_restrict_op()), 0.0);
    GKO_ASSERT_MTX_NEAR(gko::as<Csr>(d_mg_level->get_coarse_op()),
                        gko::as<Csr>(mg_level->get_coarse_op()),
                        r<value_type>::value);
}


}  // namespace
