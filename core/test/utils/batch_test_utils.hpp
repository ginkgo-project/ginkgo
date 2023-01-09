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

#ifndef GKO_CORE_TEST_UTILS_BATCH_TEST_UTILS_HPP_
#define GKO_CORE_TEST_UTILS_BATCH_TEST_UTILS_HPP_


#include <ginkgo/core/log/batch_convergence.hpp>
#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/batch_diagonal.hpp>


#include "core/log/batch_logging.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"


namespace gko {
namespace test {


template <typename ValueType>
std::unique_ptr<matrix::BatchDense<remove_complex<ValueType>>>
compute_residual_norm(const BatchLinOp* const rmtx,
                      const matrix::BatchDense<ValueType>* const x,
                      const matrix::BatchDense<ValueType>* const b)
{
    using BDense = matrix::BatchDense<ValueType>;
    using RBDense = matrix::BatchDense<remove_complex<ValueType>>;
    auto exec = rmtx->get_executor();
    const size_t nbatches = x->get_num_batch_entries();
    const int xnrhs = x->get_size().at(0)[1];
    const gko::batch_stride stride(nbatches, xnrhs);
    const gko::batch_dim<> normdim(nbatches, gko::dim<2>(1, xnrhs));

    std::unique_ptr<BDense> res = b->clone();
    auto normsr = RBDense::create(exec, normdim);
    auto alpha = gko::batch_initialize<BDense>(nbatches, {-1.0}, exec);
    auto beta = gko::batch_initialize<BDense>(nbatches, {1.0}, exec);
    rmtx->apply(alpha.get(), x, beta.get(), res.get());
    res->compute_norm2(normsr.get());
    return normsr;
}


template <typename ValueType>
struct LinSys {
    using BDense = matrix::BatchDense<ValueType>;
    using RBDense = matrix::BatchDense<remove_complex<ValueType>>;

    std::unique_ptr<BatchLinOp> mtx;
    std::unique_ptr<BDense> b;
    std::unique_ptr<RBDense> bnorm;
    std::unique_ptr<BDense> xex;
};

template <typename ValueType>
struct Result {
    using BDense = matrix::BatchDense<ValueType>;
    using RBDense = matrix::BatchDense<remove_complex<ValueType>>;

    std::shared_ptr<BDense> x;
    std::shared_ptr<RBDense> resnorm;
    gko::log::BatchLogData<ValueType> logdata;
};

template <typename ValueType>
LinSys<ValueType> get_poisson_problem(
    std::shared_ptr<const gko::ReferenceExecutor> exec, const int nrhs,
    const size_t nbatches)
{
    using BDense = matrix::BatchDense<ValueType>;
    using RBDense = matrix::BatchDense<remove_complex<ValueType>>;
    using Mtx = matrix::BatchCsr<ValueType, int>;
    LinSys<ValueType> sys;
    const int nrows = 3;
    sys.mtx = gko::test::create_poisson1d_batch<Mtx>(exec, nrows, nbatches);
    if (nrhs == 1) {
        sys.b = gko::batch_initialize<BDense>(nbatches, {-1.0, 3.0, 1.0}, exec);
        sys.xex =
            gko::batch_initialize<BDense>(nbatches, {1.0, 3.0, 2.0}, exec);
    } else if (nrhs == 2) {
        sys.b = gko::batch_initialize<BDense>(
            nbatches,
            std::initializer_list<std::initializer_list<ValueType>>{
                {-1.0, 2.0}, {3.0, -1.0}, {1.0, 0.0}},
            exec);
        sys.xex = gko::batch_initialize<BDense>(
            nbatches,
            std::initializer_list<std::initializer_list<ValueType>>{
                {1.0, 1.0}, {3.0, 0.0}, {2.0, 0.0}},
            exec);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
    const gko::batch_dim<> normdim(nbatches, gko::dim<2>(1, nrhs));
    sys.bnorm = RBDense::create(exec, normdim);
    sys.b->compute_norm2(sys.bnorm.get());
    return sys;
}


template <typename ValueType, typename SolveFunction, typename Options>
Result<ValueType> solve_poisson_uniform(
    std::shared_ptr<const Executor> d_exec, SolveFunction solve_function,
    const Options opts, const LinSys<ValueType>& sys, const int nrhs,
    std::shared_ptr<BatchLinOpFactory> prec_factory = nullptr,
    const matrix::BatchDiagonal<ValueType>* const left_scale = nullptr,
    const matrix::BatchDiagonal<ValueType>* const right_scale = nullptr)
{
    using real_type = remove_complex<ValueType>;
    using BDense = typename Result<ValueType>::BDense;
    using RBDense = typename Result<ValueType>::RBDense;
    using BDiag = gko::matrix::BatchDiagonal<ValueType>;
    using Mtx = matrix::BatchCsr<ValueType, int>;

    const size_t nbatch = sys.mtx->get_num_batch_entries();
    const int nrows = sys.mtx->get_size().at()[0];
    const gko::batch_dim<> sizes(nbatch, gko::dim<2>(nrows, nrhs));
    const gko::batch_dim<> normsizes(nbatch, gko::dim<2>(1, nrhs));

    auto exec = d_exec->get_master();
    auto orig_mtx = Mtx::create(exec);
    orig_mtx->copy_from(sys.mtx.get());
    Result<ValueType> result;
    // Initialize r to the original unscaled b
    result.x = BDense::create(exec, sizes);
    ValueType* const xvalsinit = result.x->get_values();
    for (size_t i = 0; i < nbatch * nrows * nrhs; i++) {
        xvalsinit[i] = gko::zero<ValueType>();
    }

    gko::log::BatchLogData<ValueType> logdata;
    logdata.res_norms =
        gko::matrix::BatchDense<real_type>::create(d_exec, normsizes);
    logdata.iter_counts.set_executor(d_exec);
    logdata.iter_counts.resize_and_reset(nrhs * nbatch);

    auto mtx = gko::share(Mtx::create(d_exec));
    auto b = BDense::create(d_exec);
    auto x = BDense::create(d_exec);
    mtx->copy_from(gko::lend(sys.mtx));
    b->copy_from(gko::lend(sys.b));
    x->copy_from(gko::lend(result.x));
    auto d_left = BDiag::create(d_exec);
    auto d_right = BDiag::create(d_exec);
    if (left_scale) {
        d_left->copy_from(left_scale);
    }
    if (right_scale) {
        d_right->copy_from(right_scale);
    }
    auto d_left_ptr = left_scale ? d_left.get() : nullptr;
    auto d_right_ptr = right_scale ? d_right.get() : nullptr;

    auto b_sc = BDense::create(d_exec);
    b_sc->copy_from(b.get());
    if (left_scale) {
        gko::matrix::two_sided_batch_system_transform(
            d_exec, d_left_ptr, d_right_ptr, mtx.get(), b_sc.get());
    }

    std::unique_ptr<BatchLinOp> prec;
    if (prec_factory) {
        prec = prec_factory->generate(mtx);
    } else {
        prec = nullptr;
    }

    solve_function(opts, mtx.get(), prec.get(), b_sc.get(), x.get(), logdata);

    if (left_scale) {
        d_right_ptr->apply(x.get(), x.get());
    }

    result.x->copy_from(gko::lend(x));
    auto rnorms =
        compute_residual_norm(sys.mtx.get(), result.x.get(), sys.b.get());

    result.logdata.res_norms = gko::matrix::BatchDense<real_type>::create(exec);
    result.logdata.iter_counts.set_executor(exec);
    result.logdata.res_norms->copy_from(logdata.res_norms.get());
    result.logdata.iter_counts = logdata.iter_counts;

    result.resnorm = std::move(rnorms);
    return std::move(result);
}


template <typename SolverType, typename ValueType>
Result<ValueType> solve_poisson_uniform_core(
    std::shared_ptr<const Executor> d_exec,
    const typename SolverType::Factory* const factory,
    const LinSys<ValueType>& sys, const int nrhs)
{
    using real_type = remove_complex<ValueType>;
    using BDense = typename Result<ValueType>::BDense;
    using RBDense = typename Result<ValueType>::RBDense;
    using BDiag = gko::matrix::BatchDiagonal<ValueType>;
    using Mtx = matrix::BatchCsr<ValueType, int>;

    const size_t nbatch = sys.mtx->get_num_batch_entries();
    const int nrows = sys.mtx->get_size().at()[0];
    const gko::batch_dim<> sizes(nbatch, gko::dim<2>(nrows, nrhs));
    const gko::batch_dim<> normsizes(nbatch, gko::dim<2>(1, nrhs));

    auto exec = d_exec->get_master();
    auto orig_mtx = Mtx::create(exec);
    orig_mtx->copy_from(sys.mtx.get());
    Result<ValueType> result;
    // Initialize r to the original unscaled b
    result.x = BDense::create(exec, sizes);
    ValueType* const xvalsinit = result.x->get_values();
    for (size_t i = 0; i < nbatch * nrows * nrhs; i++) {
        xvalsinit[i] = gko::zero<ValueType>();
    }

    std::shared_ptr<const gko::log::BatchConvergence<ValueType>> logger =
        gko::log::BatchConvergence<ValueType>::create(d_exec);

    std::shared_ptr<Mtx> mtx = Mtx::create(d_exec);
    auto b = BDense::create(d_exec);
    auto x = BDense::create(d_exec);
    mtx->copy_from(gko::lend(sys.mtx));
    b->copy_from(gko::lend(sys.b));
    x->copy_from(gko::lend(result.x));

    auto solver = factory->generate(mtx);

    solver->add_logger(logger);
    solver->apply(b.get(), x.get());
    solver->remove_logger(logger.get());

    result.x->copy_from(gko::lend(x));
    auto rnorms =
        compute_residual_norm(sys.mtx.get(), result.x.get(), sys.b.get());
    result.resnorm = std::move(rnorms);

    result.logdata.res_norms = gko::matrix::BatchDense<real_type>::create(exec);
    result.logdata.iter_counts.set_executor(exec);
    result.logdata.res_norms->copy_from(logger->get_residual_norm());
    result.logdata.iter_counts = logger->get_num_iterations();

    return std::move(result);
}


template <typename SolverType, typename MatrixType>
void test_solve(std::shared_ptr<const Executor> exec, const size_t nbatch,
                const int nrows, const int nrhs,
                const remove_complex<typename SolverType::value_type> res_tol,
                const int maxits,
                const typename SolverType::Factory* const factory,
                const double true_res_norm_slack_factor = 1.0,
                const bool test_logger = true)
{
    using T = typename SolverType::value_type;
    using RT = typename gko::remove_complex<T>;
    using Dense = gko::matrix::BatchDense<T>;
    using RDense = gko::matrix::BatchDense<RT>;
    using BDiag = gko::matrix::BatchDiagonal<T>;
    using Mtx = MatrixType;
    using factory_type = typename SolverType::Factory;
    std::shared_ptr<const gko::ReferenceExecutor> refexec =
        gko::ReferenceExecutor::create();
    std::shared_ptr<Mtx> ref_mtx =
        gko::test::create_poisson1d_batch<Mtx>(refexec, nrows, nbatch);
    std::shared_ptr<Mtx> mtx = Mtx::create(exec);
    mtx->copy_from(ref_mtx.get());
    auto solver = factory->generate(mtx);
    const auto s_vec_sz = gko::batch_dim<>(nbatch, gko::dim<2>(nrows, 1));
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
    if (exec != nullptr) {
        ASSERT_NO_THROW(exec->synchronize());
    }

    solver->add_logger(logger);
    solver->apply(b.get(), x.get());
    solver->remove_logger(logger.get());

    ref_x->copy_from(x.get());
    auto ref_rnorm =
        compute_residual_norm(ref_mtx.get(), ref_x.get(), ref_b.get());
    auto r_iter_array = logger->get_num_iterations();
    auto r_logged_res = logger->get_residual_norm();
    for (size_t ib = 0; ib < nbatch; ib++) {
        for (int j = 0; j < nrhs; j++) {
            if (test_logger) {
                ASSERT_GT(r_iter_array.get_const_data()[ib * nrhs + j], 0);
                ASSERT_LE(r_iter_array.get_const_data()[ib * nrhs + j],
                          maxits - 1);
                ASSERT_LE(r_logged_res->at(ib, 0, j) / ref_bnorm->at(ib, 0, j),
                          res_tol);
            }
            ASSERT_LE(ref_rnorm->at(ib, 0, j) / ref_bnorm->at(ib, 0, j),
                      true_res_norm_slack_factor * res_tol);
            if (test_logger) {
                ASSERT_LE(
                    abs(r_logged_res->at(ib, 0, j) - ref_rnorm->at(ib, 0, j)),
                    res_tol * ref_bnorm->at(ib, 0, j) *
                        (abs(true_res_norm_slack_factor - 1) +
                         10 * r<T>::value));
            }
        }
    }
}

template <typename SolverType>
void test_solve_iterations_with_scaling(
    std::shared_ptr<const Executor> exec, const size_t nbatch, const int nrows,
    const int nrhs, const typename SolverType::Factory* const factory,
    const typename SolverType::Factory* const factory_s)
{
    using value_type = typename SolverType::value_type;
    using BDense = gko::matrix::BatchDense<value_type>;
    using BDiag = gko::matrix::BatchDiagonal<value_type>;
    using Mtx = typename gko::matrix::BatchCsr<value_type>;
    auto refexec = gko::ReferenceExecutor::create();
    auto matsz = gko::batch_dim<>(nbatch, gko::dim<2>(nrows, nrows));
    auto vecsz = gko::batch_dim<>(nbatch, gko::dim<2>(nrows, 1));
    auto xex = BDense::create(refexec, vecsz);
    auto b = BDense::create(refexec, vecsz);
    std::shared_ptr<Mtx> mtx =
        gko::test::create_poisson1d_batch<Mtx>(refexec, nrows, nbatch);
    const int nnz = mtx->get_const_row_ptrs()[nrows];
    for (size_t ib = 0; ib < nbatch; ib++) {
        value_type* const values = mtx->get_values() + nnz * ib;
        for (int irow = 0; irow < nrows; irow++) {
            for (int idx = mtx->get_const_row_ptrs()[irow];
                 idx < mtx->get_const_row_ptrs()[irow + 1]; idx++) {
                if (mtx->get_const_col_idxs()[idx] == irow) {
                    values[idx] = 2.0 + irow;
                }
            }
        }
    }
    auto x = BDense::create(refexec, vecsz);
    auto x_s = BDense::create(refexec, vecsz);
    for (size_t ib = 0; ib < nbatch; ib++) {
        for (int i = 0; i < nrows; i++) {
            xex->at(ib, i, 0) = 1.0;
            x->at(ib, i, 0) = 0.0;
            x_s->at(ib, i, 0) = 0.0;
        }
    }
    mtx->apply(xex.get(), b.get());
    std::shared_ptr<Mtx> d_mtx = Mtx::create(exec);
    d_mtx->copy_from(mtx.get());
    auto d_x = BDense::create(exec);
    d_x->copy_from(x.get());
    auto d_x_s = BDense::create(exec);
    d_x_s->copy_from(x_s.get());
    auto d_b = BDense::create(exec);
    d_b->copy_from(b.get());
    std::shared_ptr<const gko::log::BatchConvergence<value_type>> logger =
        gko::log::BatchConvergence<value_type>::create(exec);
    auto solver = factory->generate(d_mtx);
    std::shared_ptr<const gko::log::BatchConvergence<value_type>> logger_s =
        gko::log::BatchConvergence<value_type>::create(exec);
    auto solver_s = factory_s->generate(d_mtx);

    solver->add_logger(logger);
    solver->apply(d_b.get(), d_x.get());
    solver->remove_logger(logger.get());
    solver_s->add_logger(logger_s);
    solver_s->apply(d_b.get(), d_x_s.get());
    solver_s->remove_logger(logger_s.get());

    x->copy_from(d_x.get());
    x_s->copy_from(d_x_s.get());
    exec->synchronize();
    auto rnorm = compute_residual_norm(mtx.get(), x.get(), b.get());
    auto rnorm_s = compute_residual_norm(mtx.get(), x_s.get(), b.get());
    for (size_t i = 0; i < nbatch; i++) {
        ASSERT_GT(rnorm->at(i, 0, 0), rnorm_s->at(i, 0, 0));
    }
}


template <typename ValueType, typename SolverType>
void compare_with_reference(std::shared_ptr<const Executor> d_exec,
                            BatchSystem<ValueType>& bsys,
                            typename SolverType::Factory* const ref_factory,
                            typename SolverType::Factory* const d_factory,
                            const double iter_tol, const double res_tol,
                            const double sol_tol)
{
    using value_type = ValueType;
    using BDense = matrix::BatchDense<ValueType>;
    auto exec = d_exec->get_master();
    const auto nbatch = bsys.A->get_num_batch_entries();
    const auto nrows = bsys.A->get_size().at(0)[0];
    const auto nrhs = bsys.b->get_size().at(0)[1];
    auto d_A = bsys.A->clone(d_exec);
    auto d_b = BDense::create(d_exec);
    d_b->copy_from(bsys.b.get());
    auto r_x = BDense::create(exec, bsys.b->get_size());
    for (size_type isys = 0; isys < nbatch; isys++) {
        for (int i = 0; i < nrows; i++) {
            r_x->at(isys, i, 0) = zero<ValueType>();
        }
    }
    auto d_x = BDense::create(d_exec);
    d_x->copy_from(r_x.get());
    // auto d_factory = SolverType::build().on(d_exec);
    // d_factory->copy_from(ref_factory);
    std::shared_ptr<const gko::log::BatchConvergence<value_type>> r_logger =
        gko::log::BatchConvergence<value_type>::create(exec);
    std::shared_ptr<const gko::log::BatchConvergence<value_type>> d_logger =
        gko::log::BatchConvergence<value_type>::create(d_exec);
    auto r_solver = ref_factory->generate(gko::give(bsys.A));
    auto d_solver = d_factory->generate(gko::give(d_A));
    r_solver->add_logger(r_logger);
    d_solver->add_logger(d_logger);

    r_solver->apply(bsys.b.get(), r_x.get());
    d_solver->apply(d_b.get(), d_x.get());
    r_solver->remove_logger(r_logger.get());
    d_solver->remove_logger(d_logger.get());

    auto r_iter_array = r_logger->get_num_iterations();
    auto r_logged_res = r_logger->get_residual_norm();
    auto d_iter_array = d_logger->get_num_iterations();
    auto d_logged_res = d_logger->get_residual_norm();
    for (size_t ib = 0; ib < nbatch; ib++) {
        for (int j = 0; j < nrhs; j++) {
            ASSERT_GT(r_iter_array.get_const_data()[ib * nrhs + j], 0);
            ASSERT_LE(r_iter_array.get_const_data()[ib * nrhs + j], 100000);
            if (r_logged_res->at(ib, 0, j) > r<value_type>::value) {
                ASSERT_LE(std::abs(r_logged_res->at(ib, 0, j) -
                                   d_logged_res->at(ib, 0, j)) /
                              r_logged_res->at(ib, 0, j),
                          res_tol);
            } else {
                ASSERT_LE(d_logged_res->at(ib, 0, j), r<value_type>::value);
            }
            if (r_iter_array.get_const_data()[ib * nrhs + j] > 0) {
                ASSERT_LE(
                    std::abs(r_iter_array.get_const_data()[ib * nrhs + j] -
                             d_iter_array.get_const_data()[ib * nrhs + j]) /
                        r_iter_array.get_const_data()[ib * nrhs + j],
                    iter_tol);
            } else {
                ASSERT_TRUE(d_iter_array.get_const_data()[ib * nrhs + j] == 0 ||
                            d_iter_array.get_const_data()[ib * nrhs + j] == 1);
            }
        }
    }
    GKO_ASSERT_BATCH_MTX_NEAR(r_x, d_x, sol_tol);
}


}  // namespace test
}  // namespace gko


#endif  // GKO_CORE_TEST_UTILS_BATCH_TEST_UTILS_HPP_
