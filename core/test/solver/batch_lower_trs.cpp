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

#include <ginkgo/core/solver/batch_lower_trs.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/batch_diagonal.hpp>
#include <ginkgo/core/preconditioner/batch_jacobi.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"


namespace {


template <typename T>
class BatchLowerTrs : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<T>;
    using Mtx = gko::matrix::BatchCsr<value_type>;
    using ubatched_mat_type = typename Mtx::unbatch_type;
    using Dense = gko::matrix::BatchDense<value_type>;
    using Solver = gko::solver::BatchLowerTrs<value_type>;

    BatchLowerTrs()
        : exec(gko::ReferenceExecutor::create()),
          mtx(get_csr_lower_matrix()),
          batchlowertrs_factory(Solver::build().on(exec)),
          solver(batchlowertrs_factory->generate(mtx))
    {}

    std::shared_ptr<const gko::Executor> exec;
    const gko::size_type nbatch = 2;
    const int nrows = 4;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<typename Solver::Factory> batchlowertrs_factory;
    std::unique_ptr<gko::BatchLinOp> solver;

    /*
        2  0  0  0
        1  2  0  0
        4  0  5  0
        0  0  7  1

        1  0  0  0
        3  4  0  0
        1  0  1  0
        0  0  4  5
    */
    std::unique_ptr<Mtx> get_csr_lower_matrix()
    {
        auto mat = Mtx::create(gko::as<const gko::ReferenceExecutor>(exec), 2,
                               gko::dim<2>(4, 4), 7);
        int* const row_ptrs = mat->get_row_ptrs();
        int* const col_idxs = mat->get_col_idxs();
        value_type* const vals = mat->get_values();

        // clang-format off
        row_ptrs[0] = 0;row_ptrs[1] = 1; row_ptrs[2] = 3;
        row_ptrs[3] = 5; row_ptrs[4] = 7;
        col_idxs[0] = 0; col_idxs[1] = 0; col_idxs[2] = 1;
        col_idxs[3] = 0;  col_idxs[4] = 2; col_idxs[5] = 2; col_idxs[6] = 3;
        vals[0] = 2.0; vals[1] = 1.0; vals[2] = 2.0; vals[3] = 4.0;
        vals[4] = 5.0; vals[5] = 7.0; vals[6] = 1.0; vals[7] = 1.0;
        vals[8] = 3.0; vals[9] = 4.0; vals[10] = 1.0; vals[11] = 1.0;
        vals[12] = 4.0; vals[13] = 5.0;
        // clang-format on
        return mat;
    }
};

TYPED_TEST_SUITE(BatchLowerTrs, gko::test::ValueTypes);


TYPED_TEST(BatchLowerTrs, FactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->batchlowertrs_factory->get_executor(), this->exec);
}


TYPED_TEST(BatchLowerTrs, FactoryCreatesCorrectSolver)
{
    using Solver = typename TestFixture::Solver;
    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_EQ(this->solver->get_size().at(i),
                  gko::dim<2>(this->nrows, this->nrows));
    }
    auto batchlowertrs_solver = static_cast<Solver*>(this->solver.get());
    ASSERT_NE(batchlowertrs_solver->get_system_matrix(), nullptr);
    ASSERT_EQ(batchlowertrs_solver->get_system_matrix(), this->mtx);
}


TYPED_TEST(BatchLowerTrs, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->batchlowertrs_factory->generate(Mtx::create(this->exec));

    copy->copy_from(this->solver.get());

    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_EQ(copy->get_size().at(i),
                  gko::dim<2>(this->nrows, this->nrows));
    }
    auto copy_mtx = static_cast<Solver*>(copy.get())->get_system_matrix();
    const auto copy_batch_mtx = static_cast<const Mtx*>(copy_mtx.get());
    GKO_ASSERT_BATCH_MTX_NEAR(this->mtx.get(), copy_batch_mtx, 0.0);
}


TYPED_TEST(BatchLowerTrs, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->batchlowertrs_factory->generate(Mtx::create(this->exec));

    copy->copy_from(std::move(this->solver));

    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_EQ(copy->get_size().at(i),
                  gko::dim<2>(this->nrows, this->nrows));
    }
    auto copy_mtx = static_cast<Solver*>(copy.get())->get_system_matrix();
    const auto copy_batch_mtx = static_cast<const Mtx*>(copy_mtx.get());
    GKO_ASSERT_BATCH_MTX_NEAR(this->mtx.get(), copy_batch_mtx, 0.0);
}


TYPED_TEST(BatchLowerTrs, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto clone = this->solver->clone();

    for (size_t i = 0; i < this->nbatch; i++) {
        ASSERT_EQ(clone->get_size().at(i),
                  gko::dim<2>(this->nrows, this->nrows));
    }
    auto clone_mtx = static_cast<Solver*>(clone.get())->get_system_matrix();
    const auto clone_batch_mtx = static_cast<const Mtx*>(clone_mtx.get());
    GKO_ASSERT_BATCH_MTX_NEAR(this->mtx.get(), clone_batch_mtx, 0.0);
}


TYPED_TEST(BatchLowerTrs, CanBeCleared)
{
    using Solver = typename TestFixture::Solver;

    this->solver->clear();

    ASSERT_EQ(this->solver->get_num_batch_entries(), 0);
    ASSERT_EQ(this->solver->get_size().get_num_batch_entries(), 0);
    auto solver_mtx =
        static_cast<Solver*>(this->solver.get())->get_system_matrix();
    ASSERT_EQ(solver_mtx, nullptr);
}


TYPED_TEST(BatchLowerTrs, CanSetCriteriaInFactory)
{
    using Solver = typename TestFixture::Solver;
    using RT = typename TestFixture::real_type;

    auto batchlowertrs_factory =
        Solver::build().with_skip_sorting(true).on(this->exec);
    auto solver = batchlowertrs_factory->generate(this->mtx);

    ASSERT_EQ(solver->get_parameters().skip_sorting, true);
}


TYPED_TEST(BatchLowerTrs, ThrowsOnRectangularMatrixInFactory)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Mtx> rectangular_mtx =
        Mtx::create(this->exec, 2, gko::dim<2>{3, 5}, 3);

    ASSERT_THROW(this->batchlowertrs_factory->generate(rectangular_mtx),
                 gko::DimensionMismatch);
}


TYPED_TEST(BatchLowerTrs, CanSetScalingOps)
{
    using value_type = typename TestFixture::value_type;
    using Solver = typename TestFixture::Solver;
    using Dense = typename TestFixture::Dense;
    using Diag = gko::matrix::BatchDiagonal<value_type>;
    auto left_scale = gko::share(Diag::create(
        this->exec,
        gko::batch_dim<>(this->nbatch, gko::dim<2>(this->nrows, this->nrows))));
    auto right_scale = gko::share(Diag::create(
        this->exec,
        gko::batch_dim<>(this->nbatch, gko::dim<2>(this->nrows, this->nrows))));
    auto batchlowertrs_factory = Solver::build()
                                     .with_left_scaling_op(left_scale)
                                     .with_right_scaling_op(right_scale)
                                     .on(this->exec);
    auto solver = batchlowertrs_factory->generate(this->mtx);

    ASSERT_EQ(solver->get_left_scaling_op(), left_scale);
    ASSERT_EQ(solver->get_right_scaling_op(), right_scale);
}


}  // namespace
