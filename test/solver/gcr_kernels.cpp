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

#include "core/solver/gcr_kernels.hpp"
#include <random>
#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/gcr.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>

#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"

namespace {

class Gcr : public CommonTestFixture {
protected:
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Gcr<value_type>;
    using norm_type = gko::remove_complex<value_type>;
    using NormVector = gko::matrix::Dense<norm_type>;
    template <typename T>
    using Dense = typename gko::matrix::Dense<T>;

    Gcr() : rand_engine(30)
    {
        mtx = gen_mtx(123, 123);
        d_mtx = gko::clone(exec, mtx);
        exec_gcr_factory =
            Solver::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(246u).on(exec),
                    gko::stop::ResidualNorm<value_type>::build()
                        .with_reduction_factor(value_type{1e-15})
                        .on(exec))
                .on(exec);

        ref_gcr_factory =
            Solver::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(246u).on(ref),
                    gko::stop::ResidualNorm<value_type>::build()
                        .with_reduction_factor(value_type{1e-15})
                        .on(ref))
                .on(ref);
    }

    template <typename ValueType = value_type, typename IndexType = index_type>
    std::unique_ptr<Dense<ValueType>> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<Dense<ValueType>>(
            num_rows, num_cols,
            std::uniform_int_distribution<IndexType>(num_cols, num_cols),
            std::normal_distribution<ValueType>(-1.0, 1.0), rand_engine, ref);
    }

    void initialize_data(int nrhs = 43)
    {
#ifdef GINKGO_FAST_TESTS
        int m = 123;
#else
        int m = 597;
#endif
        x = gen_mtx(m, nrhs);
        b = gen_mtx(m, nrhs);
        residual = gen_mtx(m, nrhs);
        A_residual = gen_mtx(m, nrhs);
        p_bases = gen_mtx(m * (gko::solver::gcr_default_krylov_dim + 1), nrhs);
        p = gen_mtx(m, nrhs);
        Ap_bases = gen_mtx(m * (gko::solver::gcr_default_krylov_dim + 1),
                           nrhs);
        Ap = gen_mtx(m, nrhs);
        alpha = gen_mtx(1, nrhs);
        Ap_norm = gen_mtx(1, nrhs);


        stop_status = gko::Array<gko::stopping_status>(ref, nrhs);
        for (size_t i = 0; i < stop_status.get_num_elems(); ++i) {
            stop_status.get_data()[i].reset();
        }
        final_iter_nums = gko::Array<gko::size_type>(ref, nrhs);
        for (size_t i = 0; i < final_iter_nums.get_num_elems(); ++i) {
            final_iter_nums.get_data()[i] = 5;
        }

        d_x = gko::clone(exec, x);
        d_b = gko::clone(exec, b);
        d_residual = gko::clone(exec, residual);
        d_A_residual = gko::clone(exec, A_residual);
        d_p_bases = gko::clone(exec, p_bases);
        d_p = gko::clone(exec, p);
        d_Ap_bases = gko::clone(exec, Ap_bases);
        d_Ap = gko::clone(exec, Ap);
        d_alpha = gko::clone(exec, alpha);
        d_Ap_norm = gko::clone(exec, Ap_norm);
        d_stop_status = gko::array<gko::stopping_status>(exec, stop_status);
        d_final_iter_nums = gko::array<gko::size_type>(exec, final_iter_nums);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;

    std::ranlux48 rand_engine;

    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> d_mtx;
    std::unique_ptr<Solver::Factory> exec_gcr_factory;
    std::unique_ptr<Solver::Factory> ref_gcr_factory;

    std::unique_ptr<Mtx> x;
    std::unique_ptr<Mtx> b;
    std::unique_ptr<Mtx> residual;
    std::unique_ptr<Mtx> A_residual;
    std::unique_ptr<Mtx> p_bases;
    std::unique_ptr<Mtx> p;
    std::unique_ptr<Mtx> Ap_bases;
    std::unique_ptr<Mtx> Ap;
    std::unique_ptr<Mtx> alpha;
    std::unique_ptr<Mtx> Ap_norm;
    gko::array<gko::stopping_status> stop_status;
    gko::array<gko::size_type> final_iter_nums;

    std::unique_ptr<Mtx> d_x;
    std::unique_ptr<Mtx> d_b;
    std::unique_ptr<Mtx> d_residual;
    std::unique_ptr<Mtx> d_A_residual;
    std::unique_ptr<Mtx> d_p_bases;
    std::unique_ptr<Mtx> d_p;
    std::unique_ptr<Mtx> d_Ap_bases;
    std::unique_ptr<Mtx> d_Ap;
    std::unique_ptr<Mtx> d_alpha;
    std::unique_ptr<Mtx> d_Ap_norm;
    gko::array<gko::stopping_status> d_stop_status;
    gko::array<gko::size_type> d_final_iter_nums;

};


TEST_F(Gcr, GcrKernelInitializeIsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::gcr::initialize(ref, b.get(), residual.get(),
                                             stop_status.get_data());
    gko::kernels::EXEC_NAMESPACE::gcr::initialize(
        exec, d_b.get(), d_residual.get(), d_stop_status.get_data());

    GKO_ASSERT_MTX_NEAR(d_residual, residual, r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(d_stop_status, stop_status);
}


TEST_F(Gcr, GcrKernelRestartIsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::gcr::restart(
        ref, residual.get(), A_residual.get(), p_bases.get(), Ap_bases.get(),
        final_iter_nums.get_data());
    gko::kernels::EXEC_NAMESPACE::gcr::restart(
        exec, d_residual.get(),
        d_A_residual.get(), d_p_bases.get(), d_Ap_bases.get(),
        d_final_iter_nums.get_data());

    GKO_ASSERT_MTX_NEAR(d_A_residual, A_residual, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_p, p, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_Ap, Ap, r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(d_final_iter_nums, final_iter_nums);
}


TEST_F(Gcr, GcrStep1IsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::gcr::step_1(ref, x.get(), residual.get(), p.get(), Ap.get(), Ap_norm.get(), alpha.get(), stop_status.get_data());
    gko::kernels::EXEC_NAMESPACE::gcr::step_1(exec, d_x.get(), d_residual.get(), d_p.get(), d_Ap.get(), d_Ap_norm.get(), d_alpha.get(), d_stop_status.get_data());

    GKO_ASSERT_MTX_NEAR(d_x, x, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_residual, residual, r<value_type>::value);
}


TEST_F(Gcr, GcrApplyOneRHSIsEquivalentToRef)
{
    int m = 123;
    int n = 1;
    auto ref_solver = ref_gcr_factory->generate(mtx);
    auto d_solver = exec_gcr_factory->generate(d_mtx);
    auto b = gen_mtx(m, n);
    auto x = gen_mtx(m, n);
    auto d_b = Mtx::create(exec);
    auto d_x = Mtx::create(exec);
    d_b->copy_from(b.get());
    d_x->copy_from(x.get());

    ref_solver->apply(b.get(), x.get());
    d_solver->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_b, b, r<value_type>::value * 1e2);
    GKO_ASSERT_MTX_NEAR(d_x, x, r<value_type>::value * 1e2);
}


TEST_F(Gcr, GcrApplyMultipleRHSIsEquivalentToRef)
{
    int m = 123;
    int n = 5;
    auto ref_solver = ref_gcr_factory->generate(mtx);
    auto omp_solver = exec_gcr_factory->generate(d_mtx);
    auto b = gen_mtx(m, n);
    auto x = gen_mtx(m, n);
    auto d_b = Mtx::create(exec);
    auto d_x = Mtx::create(exec);
    d_b->copy_from(b.get());
    d_x->copy_from(x.get());

    ref_solver->apply(b.get(), x.get());
    omp_solver->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_b, b, r<value_type>::value * 1e3);
    GKO_ASSERT_MTX_NEAR(d_x, x, r<value_type>::value * 1e3);
}


}  // namespace
