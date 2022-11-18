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

#include <ginkgo/core/solver/gcr.hpp>


#include <algorithm>
#include <limits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/core/stop/time.hpp>


#include "core/solver/gcr_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename T>
class Gcr : public ::testing::Test {
protected:
    using value_type = T;
    using rc_value_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::Dense<value_type>;
    using rc_Mtx = gko::matrix::Dense<rc_value_type>;
    using Solver = gko::solver::Gcr<value_type>;
    Gcr()
        : exec(gko::ReferenceExecutor::create()),
          stopped{},
          non_stopped{},
          mtx(gko::initialize<Mtx>(
              {{1.0, 2.0, 3.0}, {3.0, 2.0, -1.0}, {0.0, -1.0, 2}}, exec)),
          gcr_factory(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(4u).on(exec),
                      gko::stop::Time::build()
                          .with_time_limit(std::chrono::seconds(6))
                          .on(exec),
                      gko::stop::ResidualNorm<value_type>::build()
                          .with_reduction_factor(r<value_type>::value)
                          .on(exec))
                  .with_krylov_dim(3u)
                  .on(exec)),
          mtx_big(gko::initialize<Mtx>(
              {{2295.7, -764.8, 1166.5, 428.9, 291.7, -774.5},
               {2752.6, -1127.7, 1212.8, -299.1, 987.7, 786.8},
               {138.3, 78.2, 485.5, -899.9, 392.9, 1408.9},
               {-1907.1, 2106.6, 1026.0, 634.7, 194.6, -534.1},
               {-365.0, -715.8, 870.7, 67.5, 279.8, 1927.8},
               {-848.1, -280.5, -381.8, -187.1, 51.2, -176.2}},
              exec)),
          gcr_factory_big(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(100u).on(
                          exec),
                      gko::stop::ResidualNorm<value_type>::build()
                          .with_reduction_factor(r<value_type>::value)
                          .on(exec))
                  .on(exec)),
          gcr_factory_big2(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(100u).on(
                          exec),
                      gko::stop::ImplicitResidualNorm<value_type>::build()
                          .with_reduction_factor(r<value_type>::value)
                          .on(exec))
                  .on(exec)),
          mtx_medium(
              gko::initialize<Mtx>({{-86.40, 153.30, -108.90, 8.60, -61.60},
                                    {7.70, -77.00, 3.30, -149.20, 74.80},
                                    {-121.40, 37.10, 55.30, -74.20, -19.20},
                                    {-111.40, -22.60, 110.10, -106.20, 88.90},
                                    {-0.70, 111.70, 154.40, 235.00, -76.50}},
                                   exec))
    {
        auto small_size = gko::dim<2>{3, 2};
        constexpr gko::size_type small_restart{2};
        small_b = gko::initialize<Mtx>(
            {I<T>{1., 2.}, I<T>{3., 4.}, I<T>{5., 6.}}, exec);
        small_x = Mtx::create(exec, small_size);
        small_residual = Mtx::create(exec, small_size);

        stopped.converge(1, true);
        non_stopped.reset();
        small_stop = gko::array<gko::stopping_status>(exec, small_size[1]);
        std::fill_n(small_stop.get_data(), small_stop.get_num_elems(),
                    non_stopped);
        small_final_iter_nums = gko::array<gko::size_type>(exec, small_size[1]);
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::unique_ptr<Mtx> small_x;
    std::unique_ptr<Mtx> small_b;
    std::unique_ptr<Mtx> small_residual;
    gko::array<gko::size_type> small_final_iter_nums;
    gko::array<gko::stopping_status> small_stop;

    gko::stopping_status stopped;
    gko::stopping_status non_stopped;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> mtx_medium;
    std::shared_ptr<Mtx> mtx_big;
    std::unique_ptr<typename Solver::Factory> gcr_factory;
    std::unique_ptr<typename Solver::Factory> gcr_factory_big;
    std::unique_ptr<typename Solver::Factory> gcr_factory_big2;
};

TYPED_TEST_SUITE(Gcr, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Gcr, KernelInitialize)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    const T nan = std::numeric_limits<gko::remove_complex<T>>::quiet_NaN();
    this->small_residual->fill(nan);
    std::fill_n(this->small_stop.get_data(), this->small_stop.get_num_elems(),
                this->stopped);

    gko::kernels::reference::gcr::initialize(this->exec, this->small_b.get(),
                                             this->small_residual.get(),
                                             this->small_stop.get_data());

    GKO_ASSERT_MTX_NEAR(this->small_residual, this->small_b, 0);
    for (int i = 0; i < this->small_stop.get_num_elems(); ++i) {
        ASSERT_EQ(this->small_stop.get_data()[i], this->non_stopped);
    }
}


}  // namespace
