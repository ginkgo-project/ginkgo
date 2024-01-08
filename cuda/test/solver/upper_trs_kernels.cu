// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>
#include <random>


#include <cuda.h>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/triangular.hpp>


#include "core/solver/upper_trs_kernels.hpp"
#include "cuda/test/utils.hpp"


namespace {


class UpperTrs : public CudaTestFixture {
protected:
    using CsrMtx = gko::matrix::Csr<double, gko::int32>;
    using Mtx = gko::matrix::Dense<>;

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
        csr_mtx = CsrMtx::create(ref);
        mtx->convert_to(csr_mtx);
        d_csr_mtx = CsrMtx::create(exec);
        d_x = gko::clone(exec, x);
        d_csr_mtx->copy_from(csr_mtx);
        b2 = Mtx::create(ref);
        d_b2 = gko::clone(exec, b);
        b2->copy_from(b);
    }

    std::shared_ptr<Mtx> b;
    std::shared_ptr<Mtx> b2;
    std::shared_ptr<Mtx> x;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<CsrMtx> csr_mtx;
    std::shared_ptr<Mtx> d_b;
    std::shared_ptr<Mtx> d_b2;
    std::shared_ptr<Mtx> d_x;
    std::shared_ptr<CsrMtx> d_csr_mtx;
    std::default_random_engine rand_engine;
};


TEST_F(UpperTrs, CudaUpperTrsFlagCheckIsCorrect)
{
    bool trans_flag = true;
    bool expected_flag = false;

    gko::kernels::cuda::upper_trs::should_perform_transpose(exec, trans_flag);

    ASSERT_EQ(expected_flag, trans_flag);
}


TEST_F(UpperTrs, CudaSingleRhsApplySyncfreelibIsEquivalentToRef)
{
    initialize_data(50, 1);
    auto upper_trs_factory = gko::solver::UpperTrs<>::build().on(ref);
    auto d_upper_trs_factory =
        gko::solver::UpperTrs<>::build()
            .with_algorithm(gko::solver::trisolve_algorithm::syncfree)
            .on(exec);
    auto solver = upper_trs_factory->generate(csr_mtx);
    auto d_solver = d_upper_trs_factory->generate(d_csr_mtx);

    solver->apply(b2, x);
    d_solver->apply(d_b2, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
}


TEST_F(UpperTrs, CudaSingleRhsApplyIsEquivalentToRef)
{
    initialize_data(50, 1);
    auto upper_trs_factory = gko::solver::UpperTrs<>::build().on(ref);
    auto d_upper_trs_factory = gko::solver::UpperTrs<>::build().on(exec);
    auto solver = upper_trs_factory->generate(csr_mtx);
    auto d_solver = d_upper_trs_factory->generate(d_csr_mtx);

    solver->apply(b2, x);
    d_solver->apply(d_b2, d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
}


TEST_F(UpperTrs, CudaMultipleRhsApplySyncfreeIsEquivalentToRef)
{
    initialize_data(50, 3);
    auto upper_trs_factory =
        gko::solver::UpperTrs<>::build().with_num_rhs(3u).on(ref);
    auto d_upper_trs_factory =
        gko::solver::UpperTrs<>::build()
            .with_algorithm(gko::solver::trisolve_algorithm::syncfree)
            .with_num_rhs(3u)
            .on(exec);
    auto solver = upper_trs_factory->generate(csr_mtx);
    auto d_solver = d_upper_trs_factory->generate(d_csr_mtx);
    auto db2_strided = Mtx::create(exec, b->get_size(), 4);
    d_b2->convert_to(db2_strided);
    auto dx_strided = Mtx::create(exec, x->get_size(), 5);

    solver->apply(b2, x);
    d_solver->apply(db2_strided, dx_strided);

    GKO_ASSERT_MTX_NEAR(dx_strided, x, 1e-14);
}


TEST_F(UpperTrs, CudaMultipleRhsApplyIsEquivalentToRef)
{
    initialize_data(50, 3);
#if CUDA_VERSION >= 11031
#if CUDA_VERSION < 12000
    // The cuSPARSE Generic SpSM implementation uses the wrong stride here
    // so the input and output stride need to match
    auto in_stride = 4;
    auto out_stride = 4;
#else
    // The cuSPARSE 12 solver is even worse: It only works if the stride is
    // equal to the number of columns.
    auto in_stride = 3;
    auto out_stride = 3;
#endif
#else
    auto in_stride = 4;
    auto out_stride = 5;
#endif
    auto upper_trs_factory =
        gko::solver::UpperTrs<>::build().with_num_rhs(3u).on(ref);
    auto d_upper_trs_factory =
        gko::solver::UpperTrs<>::build().with_num_rhs(3u).on(exec);
    auto solver = upper_trs_factory->generate(csr_mtx);
    auto d_solver = d_upper_trs_factory->generate(d_csr_mtx);
    auto db2_strided = Mtx::create(exec, b->get_size(), in_stride);
    d_b2->convert_to(db2_strided);
    auto dx_strided = Mtx::create(exec, x->get_size(), out_stride);

    solver->apply(b2, x);
    d_solver->apply(db2_strided, dx_strided);

    GKO_ASSERT_MTX_NEAR(dx_strided, x, 1e-14);
}


TEST_F(UpperTrs, CudaApplyThrowsWithWrongNumRHS)
{
    initialize_data(50, 3);
    auto d_lower_trs_factory = gko::solver::UpperTrs<>::build().on(exec);
    auto d_solver = d_lower_trs_factory->generate(d_csr_mtx);

    ASSERT_THROW(d_solver->apply(d_b2, d_x), gko::ValueMismatch);
}


}  // namespace
