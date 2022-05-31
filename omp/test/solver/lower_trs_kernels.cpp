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

#include <ginkgo/core/solver/lower_trs.hpp>


#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/solver/lower_trs_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


class LowerTrs : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<double>;
    using Mtx2 = gko::matrix::Dense<float>;
    using CsrMtx = gko::matrix::Csr<double, gko::int32>;

    LowerTrs() : rand_engine(30), solve_struct_ref{}, solve_struct_omp{} {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        omp = gko::OmpExecutor::create();
    }

    void TearDown()
    {
        if (omp != nullptr) {
            ASSERT_NO_THROW(omp->synchronize());
        }
    }

    std::shared_ptr<Mtx> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<Mtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    std::shared_ptr<Mtx> gen_l_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_lower_triangular_matrix<Mtx>(
            num_rows, num_cols, false,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void initialize_data(int m, int n)
    {
        b = gen_mtx(m, n);
        x = gen_mtx(m, n);
        x2 = Mtx2::create(ref);
        x2->copy_from(x.get());
        t_b = b->clone();
        t_x = x->clone();
        d_b = gko::clone(omp, b);
        d_x = gko::clone(omp, x);
        d_x2 = gko::clone(omp, x2);
        dt_b = gko::clone(omp, b);
        dt_x = gko::clone(omp, x);
        mtx = gen_l_mtx(m, m);
        csr_mtx = CsrMtx::create(ref);
        mtx->convert_to(csr_mtx.get());
        d_mtx = gko::clone(omp, mtx);
        d_csr_mtx = gko::clone(omp, csr_mtx);
        b2 = Mtx2::create(ref);
        b2->copy_from(b.get());
        d_b2 = gko::clone(omp, b2);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> omp;

    std::default_random_engine rand_engine;

    std::shared_ptr<Mtx> b;
    std::shared_ptr<Mtx2> b2;
    std::shared_ptr<Mtx> x;
    std::shared_ptr<Mtx2> x2;
    std::shared_ptr<Mtx> t_b;
    std::shared_ptr<Mtx> t_x;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<CsrMtx> csr_mtx;
    std::shared_ptr<Mtx> d_b;
    std::shared_ptr<Mtx2> d_b2;
    std::shared_ptr<Mtx> d_x;
    std::shared_ptr<Mtx2> d_x2;
    std::shared_ptr<Mtx> dt_b;
    std::shared_ptr<Mtx> dt_x;
    std::shared_ptr<Mtx> d_mtx;
    std::shared_ptr<CsrMtx> d_csr_mtx;
    std::shared_ptr<gko::solver::SolveStruct> solve_struct_ref;
    std::shared_ptr<gko::solver::SolveStruct> solve_struct_omp;
};


TEST_F(LowerTrs, OmpLowerTrsFlagCheckIsCorrect)
{
    bool trans_flag = true;
    bool expected_flag = false;

    gko::kernels::omp::lower_trs::should_perform_transpose(omp, trans_flag);

    ASSERT_EQ(expected_flag, trans_flag);
}


TEST_F(LowerTrs, OmpLowerTrsGenerateIsEquivalentToRef)
{
    gko::size_type num_rhs = 1;
    gko::kernels::reference::lower_trs::generate(ref, csr_mtx.get(),
                                                 solve_struct_ref, num_rhs);
    gko::kernels::omp::lower_trs::generate(omp, d_csr_mtx.get(),
                                           solve_struct_omp, num_rhs);
}


TEST_F(LowerTrs, OmpLowerTrsSolveIsEquivalentToRef)
{
    initialize_data(59, 43);

    gko::kernels::reference::lower_trs::solve(ref, csr_mtx.get(),
                                              solve_struct_ref.get(), t_b.get(),
                                              t_x.get(), b.get(), x.get());
    gko::kernels::omp::lower_trs::solve(omp, d_csr_mtx.get(),
                                        solve_struct_omp.get(), dt_b.get(),
                                        dt_x.get(), d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
}


TEST_F(LowerTrs, OmpSingleRhsApplyClassicalIsEquivalentToRef)
{
    initialize_data(50, 1);
    auto lower_trs_factory = gko::solver::LowerTrs<>::build().on(ref);
    auto d_lower_trs_factory = gko::solver::LowerTrs<>::build().on(omp);
    d_csr_mtx->set_strategy(std::make_shared<CsrMtx::classical>());
    auto solver = lower_trs_factory->generate(csr_mtx);
    auto d_solver = d_lower_trs_factory->generate(d_csr_mtx);

    solver->apply(b.get(), x.get());
    d_solver->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
}


TEST_F(LowerTrs, OmpSingleRhsApplyIsEquivalentToRef)
{
    initialize_data(50, 1);
    auto lower_trs_factory = gko::solver::LowerTrs<>::build().on(ref);
    auto d_lower_trs_factory = gko::solver::LowerTrs<>::build().on(omp);
    auto solver = lower_trs_factory->generate(csr_mtx);
    auto d_solver = d_lower_trs_factory->generate(d_csr_mtx);

    solver->apply(b.get(), x.get());
    d_solver->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
}


TEST_F(LowerTrs, OmpSingleRhsMixedApplyIsEquivalentToRef1)
{
    initialize_data(50, 1);
    auto lower_trs_factory = gko::solver::LowerTrs<>::build().on(ref);
    auto d_lower_trs_factory = gko::solver::LowerTrs<>::build().on(omp);
    auto solver = lower_trs_factory->generate(csr_mtx);
    auto d_solver = d_lower_trs_factory->generate(d_csr_mtx);

    solver->apply(b2.get(), x2.get());
    d_solver->apply(d_b2.get(), d_x2.get());

    GKO_ASSERT_MTX_NEAR(d_x2, x2, 1e-6);
}


TEST_F(LowerTrs, OmpSingleRhsMixedApplyIsEquivalentToRef2)
{
    initialize_data(50, 1);
    auto lower_trs_factory = gko::solver::LowerTrs<>::build().on(ref);
    auto d_lower_trs_factory = gko::solver::LowerTrs<>::build().on(omp);
    auto solver = lower_trs_factory->generate(csr_mtx);
    auto d_solver = d_lower_trs_factory->generate(d_csr_mtx);

    solver->apply(b2.get(), x.get());
    d_solver->apply(d_b2.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
}


TEST_F(LowerTrs, OmpSingleRhsMixedApplyIsEquivalentToRef3)
{
    initialize_data(50, 1);
    auto lower_trs_factory = gko::solver::LowerTrs<>::build().on(ref);
    auto d_lower_trs_factory = gko::solver::LowerTrs<>::build().on(omp);
    auto solver = lower_trs_factory->generate(csr_mtx);
    auto d_solver = d_lower_trs_factory->generate(d_csr_mtx);

    solver->apply(b.get(), x2.get());
    d_solver->apply(d_b.get(), d_x2.get());

    GKO_ASSERT_MTX_NEAR(d_x2, x2, 1e-6);
}


TEST_F(LowerTrs, OmpMultipleRhsApplyClassicalIsEquivalentToRef)
{
    initialize_data(50, 3);
    auto lower_trs_factory =
        gko::solver::LowerTrs<>::build().with_num_rhs(3u).on(ref);
    auto d_lower_trs_factory =
        gko::solver::LowerTrs<>::build().with_num_rhs(3u).on(omp);
    d_csr_mtx->set_strategy(std::make_shared<CsrMtx::classical>());
    auto solver = lower_trs_factory->generate(csr_mtx);
    auto d_solver = d_lower_trs_factory->generate(d_csr_mtx);
    auto db_strided = Mtx::create(omp, b->get_size(), 4);
    d_b->convert_to(db_strided.get());
    auto dx_strided = Mtx::create(omp, x->get_size(), 5);

    solver->apply(b.get(), x.get());
    d_solver->apply(db_strided.get(), dx_strided.get());

    GKO_ASSERT_MTX_NEAR(dx_strided, x, 1e-14);
}


TEST_F(LowerTrs, OmpMultipleRhsApplyIsEquivalentToRef)
{
    initialize_data(50, 3);
    auto lower_trs_factory =
        gko::solver::LowerTrs<>::build().with_num_rhs(3u).on(ref);
    auto d_lower_trs_factory =
        gko::solver::LowerTrs<>::build().with_num_rhs(3u).on(omp);
    auto solver = lower_trs_factory->generate(csr_mtx);
    auto d_solver = d_lower_trs_factory->generate(d_csr_mtx);
    auto db_strided = Mtx::create(omp, b->get_size(), 4);
    d_b->convert_to(db_strided.get());
    auto dx_strided = Mtx::create(omp, x->get_size(), 5);

    solver->apply(b.get(), x.get());
    d_solver->apply(db_strided.get(), dx_strided.get());

    GKO_ASSERT_MTX_NEAR(dx_strided, x, 1e-14);
}


TEST_F(LowerTrs, OmpMultipleRhsMixedApplyIsEquivalentToRef1)
{
    initialize_data(50, 3);
    auto lower_trs_factory =
        gko::solver::LowerTrs<>::build().with_num_rhs(3u).on(ref);
    auto d_lower_trs_factory =
        gko::solver::LowerTrs<>::build().with_num_rhs(3u).on(omp);
    auto solver = lower_trs_factory->generate(csr_mtx);
    auto d_solver = d_lower_trs_factory->generate(d_csr_mtx);
    auto db2_strided = Mtx2::create(omp, b->get_size(), 4);
    d_b2->convert_to(db2_strided.get());
    auto dx2_strided = Mtx2::create(omp, x2->get_size(), 5);

    solver->apply(b2.get(), x2.get());
    d_solver->apply(db2_strided.get(), dx2_strided.get());

    GKO_ASSERT_MTX_NEAR(dx2_strided, x2, 1e-14);
}


TEST_F(LowerTrs, OmpMultipleRhsMixedApplyIsEquivalentToRef2)
{
    initialize_data(50, 3);
    auto lower_trs_factory =
        gko::solver::LowerTrs<>::build().with_num_rhs(3u).on(ref);
    auto d_lower_trs_factory =
        gko::solver::LowerTrs<>::build().with_num_rhs(3u).on(omp);
    auto solver = lower_trs_factory->generate(csr_mtx);
    auto d_solver = d_lower_trs_factory->generate(d_csr_mtx);
    auto db2_strided = Mtx2::create(omp, b->get_size(), 4);
    d_b2->convert_to(db2_strided.get());
    auto dx_strided = Mtx::create(omp, x->get_size(), 5);

    solver->apply(b2.get(), x.get());
    d_solver->apply(db2_strided.get(), dx_strided.get());

    GKO_ASSERT_MTX_NEAR(dx_strided, x, 1e-14);
}


TEST_F(LowerTrs, OmpMultipleRhsMixedApplyIsEquivalentToRef3)
{
    initialize_data(50, 3);
    auto lower_trs_factory =
        gko::solver::LowerTrs<>::build().with_num_rhs(3u).on(ref);
    auto d_lower_trs_factory =
        gko::solver::LowerTrs<>::build().with_num_rhs(3u).on(omp);
    auto solver = lower_trs_factory->generate(csr_mtx);
    auto d_solver = d_lower_trs_factory->generate(d_csr_mtx);
    auto db_strided = Mtx::create(omp, b->get_size(), 4);
    d_b->convert_to(db_strided.get());
    auto dx2_strided = Mtx2::create(omp, x2->get_size(), 5);

    solver->apply(b.get(), x2.get());
    d_solver->apply(db_strided.get(), dx2_strided.get());

    GKO_ASSERT_MTX_NEAR(dx2_strided, x2, 1e-14);
}


}  // namespace
