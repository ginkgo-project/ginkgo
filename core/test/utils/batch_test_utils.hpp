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

#ifndef GKO_CORE_TEST_UTILS_BATCH_TEST_UTILS_HPP_
#define GKO_CORE_TEST_UTILS_BATCH_TEST_UTILS_HPP_


#include <ginkgo/core/log/batch_convergence.hpp>
#include <ginkgo/core/solver/batch_richardson.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"


namespace gko {
namespace test {


template <typename SolverType>
void test_solve_without_scaling(
    std::shared_ptr<const Executor> exec, const size_t nbatch, const int nrows,
    const int nrhs,
    const remove_complex<typename SolverType::value_type> res_tol,
    const int maxits, const typename SolverType::Factory *const factory,
    const double true_res_norm_slack_factor = 1.0)
{
    using T = typename SolverType::value_type;
    using RT = typename gko::remove_complex<T>;
    using Dense = gko::matrix::BatchDense<T>;
    using RDense = gko::matrix::BatchDense<RT>;
    using Mtx = typename gko::matrix::BatchCsr<T>;
    using factory_type = typename SolverType::Factory;
    std::shared_ptr<gko::ReferenceExecutor> refexec =
        gko::ReferenceExecutor::create();
    std::shared_ptr<Mtx> ref_mtx =
        gko::test::create_poisson1d_batch<T>(refexec, nrows, nbatch);
    std::shared_ptr<Mtx> mtx = Mtx::create(exec);
    mtx->copy_from(ref_mtx.get());
    auto solver = factory->generate(mtx);
    std::shared_ptr<const gko::log::BatchConvergence<T>> logger =
        gko::log::BatchConvergence<T>::create(exec);
    auto ref_b = Dense::create(
        refexec, gko::batch_dim<>(nbatch, gko::dim<2>(nrows, nrhs)));
    auto ref_x = Dense::create_with_config_of(ref_b.get());
    auto ref_bnorm =
        RDense::create(refexec, gko::batch_dim<>(nbatch, gko::dim<2>(1, nrhs)));
    for (size_t ib = 0; ib < nbatch; ib++) {
        for (int j = 0; j < nrhs; j++) {
            ref_bnorm->at(ib, 0, j) = gko::zero<RT>();
            const T val = 1.0 + std::cos(ib / 2.0 - j / 4.0);
            for (int i = 0; i < nrows; i++) {
                ref_b->at(ib, i, j) = val;
                ref_x->at(ib, i, j) = 0.0;
                ref_bnorm->at(ib, 0, j) += gko::squared_norm(val);
            }
            ref_bnorm->at(ib, 0, j) = std::sqrt(ref_bnorm->at(ib, 0, j));
        }
    }
    auto x = Dense::create(exec);
    x->copy_from(ref_x.get());
    auto b = Dense::create(exec);
    b->copy_from(ref_b.get());
    auto alpha = gko::batch_initialize<Dense>(nbatch, {-1.0}, exec);
    auto beta = gko::batch_initialize<Dense>(nbatch, {1.0}, exec);
    if (exec != nullptr) {
        ASSERT_NO_THROW(exec->synchronize());
    }

    solver->add_logger(logger);
    solver->apply(b.get(), x.get());
    solver->remove_logger(logger.get());

    auto res = Dense::create(refexec, ref_b->get_size());
    res->copy_from(ref_b.get());
    ref_x->copy_from(x.get());
    auto ref_alpha = gko::batch_initialize<Dense>(nbatch, {-1.0}, refexec);
    auto ref_beta = gko::batch_initialize<Dense>(nbatch, {1.0}, refexec);
    ref_mtx->apply(ref_alpha.get(), ref_x.get(), ref_beta.get(), res.get());
    auto ref_rnorm =
        RDense::create(refexec, gko::batch_dim<>(nbatch, gko::dim<2>(1, nrhs)));
    res->compute_norm2(ref_rnorm.get());
    auto r_iter_array = logger->get_num_iterations();
    auto r_logged_res = logger->get_residual_norm();
    for (size_t ib = 0; ib < nbatch; ib++) {
        for (int j = 0; j < nrhs; j++) {
            ASSERT_GT(r_iter_array.get_const_data()[ib * nrhs + j], 0);
            ASSERT_LE(r_iter_array.get_const_data()[ib * nrhs + j], maxits - 1);
            ASSERT_LE(r_logged_res->at(ib, 0, j) / ref_bnorm->at(ib, 0, j),
                      res_tol);
            ASSERT_LE(ref_rnorm->at(ib, 0, j) / ref_bnorm->at(ib, 0, j),
                      true_res_norm_slack_factor * res_tol);
            ASSERT_LE(
                abs(r_logged_res->at(ib, 0, j) - ref_rnorm->at(ib, 0, j)),
                res_tol * ref_bnorm->at(ib, 0, j) *
                    (abs(true_res_norm_slack_factor - 1) + 10 * r<T>::value));
        }
    }
}


}  // namespace test
}  // namespace gko


#endif  // GKO_CORE_TEST_UTILS_BATCH_TEST_UTILS_HPP_
