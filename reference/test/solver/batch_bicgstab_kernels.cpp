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

#include <ginkgo/core/solver/batch_bicgstab.hpp>


#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/log/batch_logger.hpp>


#include "core/base/batch_utilities.hpp"
#include "core/matrix/batch_dense_kernels.hpp"
#include "core/solver/batch_bicgstab_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch_helpers.hpp"


template <typename T>
class BatchBicgstab : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using solver_type = gko::batch::solver::Bicgstab<value_type>;
    using Mtx = gko::batch::matrix::Dense<value_type>;
    using MVec = gko::batch::MultiVector<value_type>;
    using RealMVec = gko::batch::MultiVector<real_type>;
    using Settings = gko::kernels::batch_bicgstab::BicgstabOptions<real_type>;
    using LogData = gko::batch::log::BatchLogData<double>;
    using LinSys = gko::test::LinearSystem<Mtx>;

    BatchBicgstab()
        : exec(gko::ReferenceExecutor::create()),
          linear_system(gko::test::generate_3pt_stencil_batch_problem<Mtx>(
              exec, num_batch_items, num_rows, num_rhs))
    {
        auto executor = this->exec;
        solve_lambda = [executor](const Settings opts,
                                  const gko::batch::BatchLinOp* prec,
                                  const Mtx* mtx, const MVec* b, MVec* x,
                                  LogData& logdata) {
            gko::kernels::reference::batch_bicgstab::apply<
                typename Mtx::value_type>(executor, opts, mtx, prec, b, x,
                                          logdata);
        };
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    const real_type eps = r<value_type>::value;
    const gko::size_type num_batch_items = 2;
    const int num_rows = 3;
    const int num_rhs = 1;
    const Settings solver_settings{500, static_cast<real_type>(1e3) * eps,
                                   gko::batch::stop::ToleranceType::relative};
    LinSys linear_system;
    std::function<void(const Settings, const gko::batch::BatchLinOp*,
                       const Mtx*, const MVec*, MVec*, LogData&)>
        solve_lambda;
};

TYPED_TEST_SUITE(BatchBicgstab, gko::test::ValueTypes);


TYPED_TEST(BatchBicgstab, SolvesStencilSystem)
{
    auto res = gko::test::solve_linear_system(this->exec, this->solve_lambda,
                                              this->solver_settings,
                                              this->linear_system);

    for (size_t i = 0; i < this->num_batch_items; i++) {
        ASSERT_LE(res.res_norm->get_const_values()[i] /
                      this->linear_system.rhs_norm->get_const_values()[i],
                  this->solver_settings.residual_tol);
    }
    GKO_ASSERT_BATCH_MTX_NEAR(res.x, this->linear_system.exact_sol, this->eps);
}


TYPED_TEST(BatchBicgstab, StencilSystemLoggerIsCorrect)
{
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;

    auto res = gko::test::solve_linear_system(this->exec, this->solve_lambda,
                                              this->solver_settings,
                                              this->linear_system);

    const int ref_iters = 2;
    const int* const iter_array = res.logdata.iter_counts.get_const_data();
    const double* const res_log_array =
        res.logdata.res_norms->get_const_values();
    for (size_t i = 0; i < this->num_batch_items; i++) {
        // test logger
        GKO_ASSERT((iter_array[i] <= ref_iters + 1) &&
                   (iter_array[i] >= ref_iters - 1));
        ASSERT_LE(res_log_array[i] / this->linear_system.rhs_norm->at(i, 0, 0),
                  this->solver_settings.residual_tol);
        ASSERT_NEAR(res_log_array[i], res.res_norm->get_const_values()[i],
                    10 * this->eps);
    }
}


TYPED_TEST(BatchBicgstab, CanSolveDenseSystem)
{
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;
    using Solver = typename TestFixture::solver_type;
    using Mtx = typename TestFixture::Mtx;
    const real_type tol = 1e-5;
    const int max_iters = 1000;
    auto solver_factory =
        Solver::build()
            .with_default_max_iterations(max_iters)
            .with_default_residual_tol(tol)
            .with_tolerance_type(gko::batch::stop::ToleranceType::relative)
            .on(this->exec);
    const int num_rows = 13;
    const size_t num_batch_items = 5;
    const int num_rhs = 1;
    auto linear_system = gko::test::generate_3pt_stencil_batch_problem<Mtx>(
        this->exec, num_batch_items, num_rows, num_rhs);
    auto solver = gko::share(solver_factory->generate(linear_system.matrix));

    auto res =
        gko::test::solve_linear_system(this->exec, linear_system, solver);

    GKO_ASSERT_BATCH_MTX_NEAR(res.x, linear_system.exact_sol, this->eps);
    for (size_t i = 0; i < num_batch_items; i++) {
        ASSERT_LE(res.res_norm->get_const_values()[i] /
                      linear_system.rhs_norm->get_const_values()[i],
                  tol);
    }
}
