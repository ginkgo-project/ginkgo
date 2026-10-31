// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/upper_trs_kernels.hpp"

#include <memory>
#include <random>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/triangular.hpp>

#include "hip/test/utils.hip.hpp"


namespace {


class UpperTrs : public HipTestFixture {
protected:
    using CsrMtx = gko::matrix::Csr<double, gko::int32>;
    using Mtx = gko::matrix::Dense<double>;
    using Mtx2 = gko::matrix::Dense<float>;

    UpperTrs() : rand_engine(30) {}

    std::unique_ptr<Mtx> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<Mtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    std::unique_ptr<Mtx> gen_u_mtx(int size)
    {
        return gko::test::generate_random_upper_triangular_matrix<Mtx>(
            size, false, std::uniform_int_distribution<>(size, size),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void initialize_data(int m, int n)
    {
        mtx = gen_u_mtx(m);
        b = gen_mtx(m, n);
        x = gen_mtx(m, n);
        x2 = Mtx2::create(ref);
        x2->copy_from(x);
        csr_mtx = CsrMtx::create(ref);
        mtx->convert_to(csr_mtx);
        d_csr_mtx = CsrMtx::create(exec);
        d_x = gko::clone(exec, x);
        d_x2 = gko::clone(exec, x2);
        d_csr_mtx->copy_from(csr_mtx);
        d_b = gko::clone(exec, b);
        b2 = Mtx2::create(ref);
        b2->copy_from(b);
        d_b2 = gko::clone(exec, b2);
    }

    std::shared_ptr<Mtx> b;
    std::shared_ptr<Mtx2> b2;
    std::shared_ptr<Mtx> x;
    std::shared_ptr<Mtx2> x2;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<CsrMtx> csr_mtx;
    std::shared_ptr<Mtx> d_b;
    std::shared_ptr<Mtx2> d_b2;
    std::shared_ptr<Mtx> d_x;
    std::shared_ptr<Mtx2> d_x2;
    std::shared_ptr<CsrMtx> d_csr_mtx;
    std::default_random_engine rand_engine;
};


TEST_F(UpperTrs, HipUpperTrsFlagCheckIsCorrect)
{
    bool trans_flag = false;
    bool expected_flag = true;
    gko::kernels::hip::upper_trs::should_perform_transpose(exec, trans_flag);

    ASSERT_EQ(expected_flag, trans_flag);
}


TEST_F(UpperTrs, HipSingleRhsApplyIsEquivalentToRef)
{
    initialize_data(50, 1);
    auto solver = gko::solver::UpperTrs<>::build().on(ref)->generate(csr_mtx);
    auto d_solver =
        gko::solver::UpperTrs<>::build().on(exec)->generate(d_csr_mtx);

    solver->apply(b, x);
    d_solver->apply(d_b, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
}


TEST_F(UpperTrs, HipSingleRhsMixedApplySyncfreeIsEquivalentToRef1)
{
    initialize_data(50, 1);
    auto solver = gko::solver::UpperTrs<>::build().on(ref)->generate(csr_mtx);
    auto d_solver =
        gko::solver::UpperTrs<>::build()
            .with_algorithm(gko::solver::trisolve_algorithm::syncfree)
            .on(exec)
            ->generate(d_csr_mtx);

    solver->apply(b2, x2);
    d_solver->apply(d_b2, d_x2);

    GKO_ASSERT_MTX_NEAR(d_x2, x2, 1e-6);
}


TEST_F(UpperTrs, HipSingleRhsMixedApplySyncfreeIsEquivalentToRef2)
{
    initialize_data(50, 1);
    auto solver = gko::solver::UpperTrs<>::build().on(ref)->generate(csr_mtx);
    auto d_solver =
        gko::solver::UpperTrs<>::build()
            .with_algorithm(gko::solver::trisolve_algorithm::syncfree)
            .on(exec)
            ->generate(d_csr_mtx);

    solver->apply(b2, x);
    d_solver->apply(d_b2, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-6);
}


TEST_F(UpperTrs, HipSingleRhsMixedApplySyncfreeIsEquivalentToRef3)
{
    initialize_data(50, 1);
    auto solver = gko::solver::UpperTrs<>::build().on(ref)->generate(csr_mtx);
    auto d_solver =
        gko::solver::UpperTrs<>::build()
            .with_algorithm(gko::solver::trisolve_algorithm::syncfree)
            .on(exec)
            ->generate(d_csr_mtx);

    solver->apply(b, x2);
    d_solver->apply(d_b, d_x2);

    GKO_ASSERT_MTX_NEAR(d_x2, x2, 1e-6);
}


TEST_F(UpperTrs, HipMultipleRhsApplyIsEquivalentToRef)
{
    initialize_data(50, 3);
    auto solver =
        gko::solver::UpperTrs<>::build().with_num_rhs(3u).on(ref)->generate(
            csr_mtx);
    auto d_solver =
        gko::solver::UpperTrs<>::build().with_num_rhs(3u).on(exec)->generate(
            d_csr_mtx);
    auto db_strided = Mtx::create(exec, b->get_size(), 4);
    d_b->convert_to(db_strided);
    auto dx_strided = Mtx::create(exec, x->get_size(), 5);

    solver->apply(b, x);
    d_solver->apply(db_strided, dx_strided);

    GKO_ASSERT_MTX_NEAR(dx_strided, x, 1e-14);
}


TEST_F(UpperTrs, HipMultipleRhsMixedApplySyncfreeIsEquivalentToRef1)
{
    initialize_data(50, 3);
    auto solver =
        gko::solver::UpperTrs<>::build().with_num_rhs(3u).on(ref)->generate(
            csr_mtx);
    auto d_solver =
        gko::solver::UpperTrs<>::build()
            .with_algorithm(gko::solver::trisolve_algorithm::syncfree)
            .with_num_rhs(3u)
            .on(exec)
            ->generate(d_csr_mtx);
    auto db2_strided = Mtx2::create(exec, b->get_size(), 4);
    d_b2->convert_to(db2_strided);
    auto dx2_strided = Mtx2::create(exec, x->get_size(), 5);

    solver->apply(b2, x2);
    d_solver->apply(db2_strided, dx2_strided);

    GKO_ASSERT_MTX_NEAR(dx2_strided, x2, 1e-6);
}


TEST_F(UpperTrs, HipMultipleRhsMixedApplySyncfreeIsEquivalentToRef2)
{
    initialize_data(50, 3);
    auto solver =
        gko::solver::UpperTrs<>::build().with_num_rhs(3u).on(ref)->generate(
            csr_mtx);
    auto d_solver =
        gko::solver::UpperTrs<>::build()
            .with_algorithm(gko::solver::trisolve_algorithm::syncfree)
            .with_num_rhs(3u)
            .on(exec)
            ->generate(d_csr_mtx);
    auto db2_strided = Mtx2::create(exec, b->get_size(), 4);
    d_b2->convert_to(db2_strided);
    auto dx_strided = Mtx::create(exec, x->get_size(), 5);

    solver->apply(b2, x);
    d_solver->apply(db2_strided, dx_strided);

    GKO_ASSERT_MTX_NEAR(dx_strided, x, 1e-6);
}


TEST_F(UpperTrs, HipMultipleRhsMixedApplySyncfreeIsEquivalentToRef3)
{
    initialize_data(50, 3);
    auto solver =
        gko::solver::UpperTrs<>::build().with_num_rhs(3u).on(ref)->generate(
            csr_mtx);
    auto d_solver =
        gko::solver::UpperTrs<>::build()
            .with_algorithm(gko::solver::trisolve_algorithm::syncfree)
            .with_num_rhs(3u)
            .on(exec)
            ->generate(d_csr_mtx);
    auto db_strided = Mtx::create(exec, b->get_size(), 4);
    d_b->convert_to(db_strided);
    auto dx2_strided = Mtx2::create(exec, x->get_size(), 5);

    solver->apply(b, x2);
    d_solver->apply(db_strided, dx2_strided);

    GKO_ASSERT_MTX_NEAR(dx2_strided, x2, 1e-6);
}


}  // namespace
