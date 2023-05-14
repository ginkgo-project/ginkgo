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


#include <ginkgo/ginkgo.hpp>


#include <fstream>
#include <ginkgo/core/solver/chebyshev.hpp>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

#define ENABLE_PROFILE 0

#if ENABLE_PROFILE
#include <cuda_profiler_api.h>
#endif

template <typename ValueType>
std::shared_ptr<typename gko::solver::Ir<ValueType>::Factory> generate_sj_ir(
    std::shared_ptr<const gko::Executor> exec, gko::size_type iteration,
    gko::uint32 max_block = 1u)
{
    using IndexType = int;
    using ir = gko::solver::Ir<ValueType>;
    using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;
    return gko::share(
        ir::build()
            .with_solver(bj::build()
                             .with_max_block_size(max_block)
                             .with_skip_sorting(true)
                             .on(exec))
            .with_relaxation_factor(static_cast<ValueType>(0.9))
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(iteration).on(
                    exec))
            .on(exec));
}

template <typename ValueType>
std::shared_ptr<typename gko::solver::Chebyshev<ValueType>::Factory>
generate_l1sj_cheb(std::shared_ptr<const gko::Executor> exec,
                   gko::size_type iteration)
{
    using IndexType = int;
    using cheb = gko::solver::Chebyshev<ValueType>;
    using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;
    return gko::share(
        cheb::build()
            .with_solver(bj::build()
                             .with_l1(true)
                             .with_max_block_size(1u)
                             .with_skip_sorting(true)
                             .on(exec))
            .with_num_keep(2)
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(iteration).on(
                    exec))
            .on(exec));
}

template <typename ValueType, typename WorkingType, typename MultigridType>
std::shared_ptr<typename gko::multigrid::Pgm<ValueType, int, WorkingType,
                                             MultigridType>::Factory>
generate_pgm(std::shared_ptr<const gko::Executor> exec, double scalar)
{
    return gko::share(
        gko::multigrid::Pgm<ValueType, int, WorkingType, MultigridType>::build()
            .with_deterministic(true)
            .with_skip_sorting(true)
            .with_scalar(static_cast<ValueType>(scalar))
            .on(exec));
}

int main(int argc, char* argv[])
{
    // Some shortcuts
    using ValueType = double;
    using MixedType = float;
    using MixedType2 = gko::half;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using mg = gko::solver::Multigrid;
    using cg = gko::solver::Cg<ValueType>;

    // Print version information
    std::cout << gko::version_info::get() << std::endl;
    std::cout << "./exec executor mixed_mode num_level cycle_mode mg_mode "
                 "sm_mode scale A [b]"
              << std::endl;

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    // Figure out where to run the code
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [] {
                 return gko::CudaExecutor::create(0, gko::OmpExecutor::create(),
                                                  true);
             }},
            {"hip",
             [] {
                 return gko::HipExecutor::create(0, gko::OmpExecutor::create(),
                                                 true);
             }},
            {"dpcpp",
             [] {
                 return gko::DpcppExecutor::create(
                     0, gko::ReferenceExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string)();  // throws if not valid
    const int mixed_mode = argc >= 3 ? std::atoi(argv[2]) : 1;
    const unsigned num_max_levels = argc >= 4 ? std::atoi(argv[3]) : 10u;
    const std::string cycle_mode = argc >= 5 ? argv[4] : "v";
    const std::string mg_mode =
        argc >= 6 ? argv[5] : "solver";  // or cg (or preconditioner)
    if (cycle_mode != "v" && cycle_mode != "w" && cycle_mode != "f") {
        std::cout << "cycle_mode should be v, w, f";
        return -1;
    }
    if (mg_mode != "solver" && mg_mode != "cg" && mg_mode != "preconditioner") {
        std::cout << "mg_mode should be solver, cg, preconditioner";
        return -1;
    }
    const std::string sm_mode = argc >= 7 ? argv[6] : "jacobi";
    const bool scale = argc >= 8 ? (std::atoi(argv[7]) != 0) : false;
    const std::string A_file = argc >= 9 ? argv[8] : "data/A.mtx";

    const std::string b_file = argc >= 10 ? argv[9] : "ones";
    int switch_to_single = mixed_mode % 400 / 10;
    int switch_to_half = mixed_mode % 400 % 10;
    std::cout << "mixed mode: " << mixed_mode << std::endl;
    // clang-format off
    if (mixed_mode == 0) {
        std::cout << "         all leverls are double" << std::endl;
    } else if (mixed_mode == 1) {
        std::cout << "          first level is double"<< std::endl
                  << ", the rest of levels are float" << std::endl;
    } else if (mixed_mode == 2) {
        std::cout << "          first level is double" << std::endl
                  << ",        second level is float" << std::endl
                  << ", the rest of levels are half" << std::endl;
    } else if (mixed_mode == 3) {
        std::cout << "          first level is double" << std::endl
                  << ", the rest of levels are half" << std::endl;
    } else if (mixed_mode == 4) {
        std::cout << "          all level is float" << std::endl;
    } else if (mixed_mode == 5) {
        std::cout << "          first level is float" << std::endl
                  << ", the rest of levels are half" << std::endl;
    } else if (mixed_mode == 6) {
        std::cout << "          all level is half" << std::endl;
    } else if (mixed_mode >= 400 && mixed_mode <= 499) {
        if (switch_to_single > switch_to_half || switch_to_single < 1) {
            std::cout << "switch to single should be eariler than switch to half" << std::endl;
            std::cout << "switch to single should >= 1" << std::endl;
            std::exit(1);
        }
        std::cout << "0 ~ " << switch_to_single - 1 << " levels use double" << std::endl
                  << switch_to_single << " ~ " << switch_to_half - 1 << " levels use single" << std::endl
                  << "rest use half" << std::endl;
    } else if (mixed_mode >= 40 && mixed_mode <= 49) {
        switch_to_single = mixed_mode%40%10;
        if (switch_to_single < 1) {
            std::cout << "switch to single should >= 1" << std::endl;
            std::exit(1);
        }
        std::cout << "0 ~ " << switch_to_single - 1 << " levels use double" << std::endl
                  << "rest use float" << std::endl;
    } else if (mixed_mode < 0) {
        std::cout << "complex" << std::endl;
    }else {
        std::exit(1);
    }
    std::cout << "The maxium number of levels: " << num_max_levels << std::endl;
    std::cout << "cycle mode: " << cycle_mode << std::endl;
    std::cout << "mg mode: " << mg_mode << std::endl;
    std::cout << "sm mode: " << sm_mode << std::endl;
    std::cout << "scale: " << scale << std::endl;
    std::cout << "A: " << A_file << std::endl;
    std::cout << "b: " << b_file << std::endl;
    gko::solver::multigrid::cycle cycle;
    if (cycle_mode == "v") {
        cycle = gko::solver::multigrid::cycle::v;
    } else if (cycle_mode == "f") {
        cycle = gko::solver::multigrid::cycle::f;
    } if (cycle_mode == "w") {
        cycle = gko::solver::multigrid::cycle::w;
    }

    // clang-format on
    // Read data
    // auto A = share(gko::read<mtx>(std::ifstream(A_file), exec));
    auto f = std::ifstream(A_file);
    auto A_host = gko::share(
        mtx::create(exec->get_master(), std::make_shared<mtx::classical>()));
    auto mat_data =
        gko::read_raw<typename mtx::value_type, typename mtx::index_type>(f);
    // mat_data.remove_zeros();
    A_host->read(mat_data);
    // scaling
    // auto A = B;
    // auto A = gko::share(mtx::create(exec, B->get_size(), 0,
    //                                 std::make_shared<mtx::classical>()));
    auto diag = A_host->extract_diagonal()->clone(exec->get_master());
    ValueType max_diag = 0;
    for (int i = 0; i < diag->get_size()[0]; i++) {
        if (std::abs(diag->get_values()[i]) > max_diag) {
            max_diag = std::abs(diag->get_values()[i]);
        }
    }
    std::vector<ValueType> row_max(A_host->get_size()[0], 0);
    std::vector<ValueType> col_max(A_host->get_size()[0], 0);
    double scalar = 1.0;
    // using max of row/col
    if (scale) {
        for (int row = 0; row < A_host->get_size()[0]; row++) {
            for (auto i = A_host->get_row_ptrs()[row];
                 i < A_host->get_row_ptrs()[row + 1]; i++) {
                auto col = A_host->get_col_idxs()[i];
                auto val = std::abs(A_host->get_values()[i]);
                row_max.at(row) = std::max(row_max.at(row), val);
                col_max.at(col) = std::max(col_max.at(col), val);
            }
            scalar = 0.5;
        }

        std::cout << "abs max of row max: "
                  << *std::max_element(row_max.begin(), row_max.end())
                  << std::endl;
        std::cout << "abs max of col max: "
                  << *std::max_element(col_max.begin(), col_max.end())
                  << std::endl;

        for (int row = 0; row < A_host->get_size()[0]; row++) {
            for (auto i = A_host->get_row_ptrs()[row];
                 i < A_host->get_row_ptrs()[row + 1]; i++) {
                auto col = A_host->get_col_idxs()[i];

                A_host->get_values()[i] /= std::sqrt(row_max.at(row));
                A_host->get_values()[i] /= std::sqrt(col_max.at(col));
            }
        }
    }
    auto A = gko::share(A_host->clone(exec));
    // std::cout << "max_diag " << max_diag << " scaling "
    //           << 65504 / 10.0 / max_diag << std::endl;
    // auto scale_host =
    //     gko::initialize<vec>({65504 / 10.0 / max_diag}, exec->get_master());
    // auto scale = gko::initialize<vec>({65504 / 10.0 / max_diag}, exec);
    // A->scale(scale.get());
    // Create RHS as 1 and initial guess as 0
    gko::size_type size = A->get_size()[0];
    auto host_x = vec::create(exec->get_master(), gko::dim<2>(size, 1));
    auto host_b = share(vec::create(exec->get_master(), gko::dim<2>(size, 1)));
    for (auto i = 0; i < size; i++) {
        host_x->at(i, 0) = 0.;
    }
    if (b_file == "ones") {
        for (auto i = 0; i < size; i++) {
            host_b->at(i, 0) = 1.;
        }
    } else {
        host_b = share(gko::read<vec>(std::ifstream(b_file), exec));
    }
    auto x = vec::create(exec);
    auto b = vec::create(exec);
    // host_b->scale(scale_host.get());
    x->copy_from(host_x.get());
    b->copy_from(host_b.get());
    // b->scale(scale.get());
    // auto b = vec::create(exec, bb->get_size());
    // Scaling
    // b = bb;
    // diag->inverse_apply(bb.get(), b.get());

    // Calculate initial residual by overwriting b
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto initres = gko::initialize<vec>({0.0}, exec);
    A->apply(one, x, neg_one, b);
    b->compute_norm2(initres);

    // copy b again
    b->copy_from(host_b.get());

    // Prepare the stopping criteria
    const gko::remove_complex<ValueType> tolerance = 1e-9;
    const unsigned mg_iter = mg_mode == "solver" ? 100u : 1u;
    const gko::solver::initial_guess_mode initial_mode =
        mg_mode == "solver" ? gko::solver::initial_guess_mode::provided
                            : gko::solver::initial_guess_mode::zero;
    auto iter_stop = gko::share(
        gko::stop::Iteration::build().with_max_iters(mg_iter).on(exec));
    auto tol_stop = gko::share(gko::stop::ResidualNorm<ValueType>::build()
                                   .with_baseline(gko::stop::mode::absolute)
                                   .with_reduction_factor(tolerance)
                                   .on(exec));
    auto cg_iter = 700u;
#if ENABLE_PROFILE
    cg_iter = 3u;
#endif
    auto cg_iter_stop = gko::share(
        gko::stop::Iteration::build().with_max_iters(cg_iter).on(exec));
    auto cg_tol_stop =
        gko::share(gko::stop::ImplicitResidualNorm<ValueType>::build()
                       .with_baseline(gko::stop::mode::initial_resnorm)
                       .with_reduction_factor(1e-12)
                       .on(exec));

    std::vector<std::shared_ptr<const gko::stop::CriterionFactory>> criterion;

    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create();
    criterion.push_back(iter_stop);
    if (mg_mode == "solver") {
        iter_stop->add_logger(logger);
        tol_stop->add_logger(logger);
        criterion.push_back(tol_stop);
    } else if (mg_mode == "preconditioner") {
        iter_stop->add_logger(logger);
    } else if (mg_mode == "cg") {
        cg_iter_stop->add_logger(logger);
        cg_tol_stop->add_logger(logger);
    }

    // Create smoother factory (ir with bj)
    std::shared_ptr<gko::LinOpFactory> smoother_gen_d, smoother_gen_f,
        smoother_gen_h;
    std::shared_ptr<gko::LinOpFactory> coarsest_solver_gen_d,
        coarsest_solver_gen_f, coarsest_solver_gen_h;
    if (sm_mode == "jacobi") {
        smoother_gen_d = generate_sj_ir<double>(exec, 1u, 1u);
        smoother_gen_f = generate_sj_ir<float>(exec, 1u, 1u);
        smoother_gen_h = generate_sj_ir<gko::half>(exec, 1u, 1u);
        coarsest_solver_gen_d = generate_sj_ir<double>(exec, 4u, 1u);
        coarsest_solver_gen_f = generate_sj_ir<float>(exec, 4u, 1u);
        coarsest_solver_gen_h = generate_sj_ir<gko::half>(exec, 4u, 1u);
    } else if (sm_mode == "bj") {
        // 32 block_size
        smoother_gen_d = generate_sj_ir<double>(exec, 1u, 32u);
        smoother_gen_f = generate_sj_ir<float>(exec, 1u, 32u);
        smoother_gen_h = generate_sj_ir<gko::half>(exec, 1u, 32u);
        coarsest_solver_gen_d = generate_sj_ir<double>(exec, 4u, 32u);
        coarsest_solver_gen_f = generate_sj_ir<float>(exec, 4u, 32u);
        coarsest_solver_gen_h = generate_sj_ir<gko::half>(exec, 4u, 32u);
    } else if (sm_mode == "l1cheyb") {
        smoother_gen_d = generate_l1sj_cheb<double>(exec, 2u);
        smoother_gen_f = generate_l1sj_cheb<float>(exec, 2u);
        smoother_gen_h = generate_l1sj_cheb<gko::half>(exec, 2u);
        coarsest_solver_gen_d = generate_l1sj_cheb<double>(exec, 2u);
        coarsest_solver_gen_f = generate_l1sj_cheb<float>(exec, 2u);
        coarsest_solver_gen_h = generate_l1sj_cheb<gko::half>(exec, 2u);
    } else {
        std::cout << "not supported sm_mode " << sm_mode << std::endl;
        std::exit(1);
    }


    // Create RestrictProlong factory<coarse, fine, working> (always use double
    // as fine matrix in generation)
    auto mg_level_gen_dd = generate_pgm<double, double, double>(exec, scalar);
    auto mg_level_gen_fd = generate_pgm<float, double, double>(exec, scalar);
    auto mg_level_gen_ff = generate_pgm<float, double, float>(exec, scalar);
    auto mg_level_gen_hd =
        generate_pgm<gko::half, double, double>(exec, scalar);
    auto mg_level_gen_hf = generate_pgm<gko::half, double, float>(exec, scalar);
    auto mg_level_gen_hh =
        generate_pgm<gko::half, double, gko::half>(exec, scalar);
    // Create multigrid factory
    std::shared_ptr<gko::solver::Multigrid::Factory> multigrid_gen;
    if (mixed_mode == 0) {
        multigrid_gen = mg::build()
                            .with_max_levels(num_max_levels)
                            .with_min_coarse_rows(64u)
                            .with_pre_smoother(smoother_gen_d)
                            .with_post_uses_pre(true)
                            .with_mg_level(mg_level_gen_dd)
                            .with_coarsest_solver(coarsest_solver_gen_d)
                            .with_criteria(criterion)
                            .with_cycle(cycle)
                            .with_default_initial_guess(initial_mode)
                            .on(exec);
    } else if (mixed_mode == 1) {
        multigrid_gen =
            mg::build()
                .with_max_levels(num_max_levels)
                .with_min_coarse_rows(64u)
                .with_pre_smoother(smoother_gen_d, smoother_gen_f)
                .with_post_uses_pre(true)
                .with_mg_level(mg_level_gen_dd, mg_level_gen_ff)
                .with_level_selector([](const gko::size_type level,
                                        const gko::LinOp*) -> gko::size_type {
                    // The first (index 0) level will use the first
                    // mg_level_gen, smoother_gen which are the factories with
                    // ValueType. The rest of levels (>= 1) will use the second
                    // (index 1) mg_level_gen2 and smoother_gen2 which use the
                    // MixedType. The rest of levels will use different type
                    // than the normal multigrid.
                    return level >= 1 ? 1 : level;
                })
                .with_coarsest_solver(coarsest_solver_gen_f)
                .with_criteria(criterion)
                .with_cycle(cycle)
                .with_default_initial_guess(initial_mode)
                .on(exec);
    } else if (mixed_mode == 2) {
        multigrid_gen =
            mg::build()
                .with_max_levels(num_max_levels)
                .with_min_coarse_rows(64u)
                // smoother using float
                .with_pre_smoother(smoother_gen_d, smoother_gen_f,
                                   smoother_gen_h)
                // .with_post_uses_pre(false)
                // .with_post_smoother(post_smoother_gen, post_smoother_gen,
                //                     post_smoother_gen)
                .with_mg_level(mg_level_gen_dd, mg_level_gen_ff,
                               mg_level_gen_hh)
                .with_level_selector([](const gko::size_type level,
                                        const gko::LinOp*) -> gko::size_type {
                    // The first (index 0) level will use the first
                    // mg_level_gen, smoother_gen which are the factories with
                    // ValueType. The rest of levels (>= 1) will use the second
                    // (index 1) mg_level_gen2 and smoother_gen2 which use the
                    // MixedType. The rest of levels will use different type
                    // than the normal multigrid.
                    return level >= 2 ? 2 : level;
                })
                .with_coarsest_solver(coarsest_solver_gen_h)
                .with_criteria(criterion)
                .with_cycle(cycle)
                .with_default_initial_guess(initial_mode)
                .on(exec);
    } else if (mixed_mode == 3) {
        multigrid_gen =
            mg::build()
                .with_max_levels(num_max_levels)
                .with_min_coarse_rows(64u)
                .with_pre_smoother(smoother_gen_d, smoother_gen_h)
                .with_post_uses_pre(true)
                .with_mg_level(mg_level_gen_dd, mg_level_gen_hh)
                .with_level_selector([](const gko::size_type level,
                                        const gko::LinOp*) -> gko::size_type {
                    // The first (index 0) level will use the first
                    // mg_level_gen, smoother_gen which are the factories with
                    // ValueType. The rest of levels (>= 1) will use the second
                    // (index 1) mg_level_gen2 and smoother_gen2 which use the
                    // MixedType. The rest of levels will use different type
                    // than the normal multigrid.
                    return level >= 1 ? 1 : level;
                })
                .with_coarsest_solver(coarsest_solver_gen_h)
                .with_criteria(criterion)
                .with_cycle(cycle)
                .with_default_initial_guess(initial_mode)
                .on(exec);
    } else if (mixed_mode == 4) {
        multigrid_gen = mg::build()
                            .with_max_levels(num_max_levels)
                            .with_min_coarse_rows(64u)
                            .with_pre_smoother(smoother_gen_f)
                            .with_post_uses_pre(true)
                            .with_mg_level(mg_level_gen_ff)
                            .with_coarsest_solver(coarsest_solver_gen_f)
                            .with_criteria(criterion)
                            .with_cycle(cycle)
                            .with_default_initial_guess(initial_mode)
                            .on(exec);
    } else if (mixed_mode == 5) {
        multigrid_gen =
            mg::build()
                .with_max_levels(num_max_levels)
                .with_min_coarse_rows(64u)
                .with_pre_smoother(smoother_gen_f, smoother_gen_h)
                .with_post_uses_pre(true)
                .with_mg_level(mg_level_gen_ff, mg_level_gen_hh)
                .with_level_selector([](const gko::size_type level,
                                        const gko::LinOp*) -> gko::size_type {
                    return level >= 1 ? 1 : level;
                })
                .with_coarsest_solver(coarsest_solver_gen_h)
                .with_criteria(criterion)
                .with_cycle(cycle)
                .with_default_initial_guess(initial_mode)
                .on(exec);
    } else if (mixed_mode == 6) {
        multigrid_gen = mg::build()
                            .with_max_levels(num_max_levels)
                            .with_min_coarse_rows(64u)
                            .with_pre_smoother(smoother_gen_h)
                            .with_post_uses_pre(true)
                            .with_mg_level(mg_level_gen_hh)
                            .with_coarsest_solver(coarsest_solver_gen_h)
                            .with_criteria(criterion)
                            .with_cycle(cycle)
                            .with_default_initial_guess(initial_mode)
                            .on(exec);
    } else if (mixed_mode >= 400 && mixed_mode <= 499) {
        multigrid_gen =
            mg::build()
                .with_max_levels(num_max_levels)
                .with_min_coarse_rows(64u)
                .with_pre_smoother(smoother_gen_d, smoother_gen_f,
                                   smoother_gen_h)
                // .with_post_uses_pre(false)
                // .with_post_smoother(post_smoother_gen, post_smoother_gen,
                //                     post_smoother_gen)
                .with_mg_level(mg_level_gen_dd, mg_level_gen_ff,
                               mg_level_gen_hh)
                .with_level_selector([=](const gko::size_type level,
                                         const gko::LinOp*) -> gko::size_type {
                    if (level >= switch_to_half) {
                        return 2;
                    }
                    if (level >= switch_to_single) {
                        return 1;
                    }
                    return 0;
                })
                .with_coarsest_solver(coarsest_solver_gen_h)
                .with_criteria(criterion)
                .with_cycle(cycle)
                .with_default_initial_guess(initial_mode)
                .on(exec);
    } else if (mixed_mode >= 40 && mixed_mode <= 49) {
        multigrid_gen =
            mg::build()
                .with_max_levels(num_max_levels)
                .with_min_coarse_rows(64u)
                .with_pre_smoother(smoother_gen_d, smoother_gen_f)
                .with_post_uses_pre(true)
                .with_mg_level(mg_level_gen_dd, mg_level_gen_ff)
                .with_level_selector([=](const gko::size_type level,
                                         const gko::LinOp*) -> gko::size_type {
                    if (level >= switch_to_single) {
                        return 1;
                    }
                    return 0;
                })
                .with_coarsest_solver(coarsest_solver_gen_f)
                .with_criteria(criterion)
                .with_cycle(cycle)
                .with_default_initial_guess(initial_mode)
                .on(exec);
    } else if (mixed_mode == -11) {
        // V: d-s
        // M: s
        multigrid_gen =
            mg::build()
                .with_max_levels(num_max_levels)
                .with_min_coarse_rows(64u)
                // input and output are double
                // first smoother needs to be double to avoid casting copy
                .with_pre_smoother(smoother_gen_d, smoother_gen_f)
                .with_post_uses_pre(true)
                .with_mg_level(mg_level_gen_ff, mg_level_gen_ff)
                .with_level_selector([](const gko::size_type level,
                                        const gko::LinOp*) -> gko::size_type {
                    return level >= 1 ? 1 : level;
                })
                .with_coarsest_solver(coarsest_solver_gen_f)
                .with_criteria(criterion)
                .with_cycle(cycle)
                .with_default_initial_guess(initial_mode)
                .on(exec);
    } else if (mixed_mode == -12) {
        // V: d-s
        // M: h
        multigrid_gen =
            mg::build()
                .with_max_levels(num_max_levels)
                .with_min_coarse_rows(64u)
                // input and output are double
                // first smoother needs to be double to avoid casting copy
                .with_pre_smoother(smoother_gen_d, smoother_gen_f)
                .with_post_uses_pre(true)
                .with_mg_level(mg_level_gen_hf, mg_level_gen_hf)
                .with_level_selector([](const gko::size_type level,
                                        const gko::LinOp*) -> gko::size_type {
                    return level >= 1 ? 1 : level;
                })
                .with_coarsest_solver(coarsest_solver_gen_f)
                .with_criteria(criterion)
                .with_cycle(cycle)
                .with_default_initial_guess(initial_mode)
                .on(exec);
    } else if (mixed_mode == -13) {
        // V: d-s
        // M: d-s-h
        multigrid_gen =
            mg::build()
                .with_max_levels(num_max_levels)
                .with_min_coarse_rows(64u)
                // smoother using float
                .with_pre_smoother(smoother_gen_d, smoother_gen_f,
                                   smoother_gen_f)
                .with_mg_level(mg_level_gen_dd, mg_level_gen_ff,
                               mg_level_gen_hf)
                .with_level_selector([](const gko::size_type level,
                                        const gko::LinOp*) -> gko::size_type {
                    return level >= 2 ? 2 : level;
                })
                .with_coarsest_solver(coarsest_solver_gen_f)
                .with_criteria(criterion)
                .with_cycle(cycle)
                .with_default_initial_guess(initial_mode)
                .on(exec);
    } else if (mixed_mode == -21) {
        // V: d
        // M: s
        multigrid_gen = mg::build()
                            .with_max_levels(num_max_levels)
                            .with_min_coarse_rows(64u)
                            // smoother using float
                            .with_pre_smoother(smoother_gen_d)
                            .with_mg_level(mg_level_gen_fd)
                            .with_coarsest_solver(coarsest_solver_gen_d)
                            .with_criteria(criterion)
                            .with_cycle(cycle)
                            .with_default_initial_guess(initial_mode)
                            .on(exec);
    } else if (mixed_mode == -22) {
        // V: d
        // M: h
        multigrid_gen = mg::build()
                            .with_max_levels(num_max_levels)
                            .with_min_coarse_rows(64u)
                            // smoother using float
                            .with_pre_smoother(smoother_gen_d)
                            .with_mg_level(mg_level_gen_hd)
                            .with_coarsest_solver(coarsest_solver_gen_d)
                            .with_criteria(criterion)
                            .with_cycle(cycle)
                            .with_default_initial_guess(initial_mode)
                            .on(exec);
    } else if (mixed_mode == -23) {
        // V: d
        // M: d-s-h
        multigrid_gen =
            mg::build()
                .with_max_levels(num_max_levels)
                .with_min_coarse_rows(64u)
                // smoother using float
                .with_pre_smoother(smoother_gen_d, smoother_gen_d,
                                   smoother_gen_d)
                .with_mg_level(mg_level_gen_dd, mg_level_gen_fd,
                               mg_level_gen_hd)
                .with_level_selector([](const gko::size_type level,
                                        const gko::LinOp*) -> gko::size_type {
                    return level >= 2 ? 2 : level;
                })
                .with_coarsest_solver(coarsest_solver_gen_d)
                .with_criteria(criterion)
                .with_cycle(cycle)
                .with_default_initial_guess(initial_mode)
                .on(exec);
    }
    std::chrono::nanoseconds gen_time(0);
    auto gen_tic = std::chrono::steady_clock::now();
    // auto solver = solver_gen->generate(A);
    auto solver = gko::share(multigrid_gen->generate(A));
    exec->synchronize();
    auto gen_toc = std::chrono::steady_clock::now();
    gen_time +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(gen_toc - gen_tic);
    auto cg_solver = gko::share(cg::build()
                                    .with_generated_preconditioner(solver)
                                    .with_criteria(cg_iter_stop, cg_tol_stop)
                                    .on(exec)
                                    ->generate(A));


    auto mg_level_list = solver->get_mg_level_list();
    if (mixed_mode >= 0) {
        std::cout << "Level: " << mg_level_list.size() << std::endl;
        if (mixed_mode <= 3) {
            int prev_n = solver->get_system_matrix()->get_size()[0];
            int prev_nnz = gko::as<mtx>(solver->get_system_matrix())
                               ->get_num_stored_elements();
            int total_n = prev_n;
            int total_nnz = prev_nnz;
            std::cout << "0, " << prev_n << ", " << prev_nnz
                      << ", prev_n(%), prev_nnz(%), total_n(%), total_nnz(%)"
                      << std::endl;

            for (int i = 1; i < mg_level_list.size(); i++) {
                auto op = mg_level_list.at(i)->get_fine_op();
                int n = op->get_size()[0];
                int num_stored_elements = 0;
                if ((mixed_mode == 2 && i >= 2) ||
                    (mixed_mode == 3 && i >= 1)) {
                    auto csr =
                        gko::as<gko::matrix::Csr<MixedType2, IndexType>>(op);
                    num_stored_elements = csr->get_num_stored_elements();
                    // std::string file = "matrix_" + std::to_string(i) +
                    // ".mtx"; write(std::ofstream(file), csr);
                } else if ((mixed_mode == 1 && i >= 1) ||
                           (mixed_mode == 2 && i == 1)) {
                    auto csr =
                        gko::as<gko::matrix::Csr<MixedType, IndexType>>(op);
                    num_stored_elements = csr->get_num_stored_elements();
                } else {
                    auto csr = gko::as<mtx>(op);
                    num_stored_elements = csr->get_num_stored_elements();
                }
                std::cout << i << ", " << n << ", " << num_stored_elements
                          << ", " << float(n) / prev_n << ", "
                          << float(num_stored_elements) / prev_nnz << ", "
                          << float(n) / total_n << ", "
                          << float(num_stored_elements) / total_nnz
                          << std::endl;
                prev_n = n;
                prev_nnz = num_stored_elements;
            }
            {
                auto op =
                    mg_level_list.at(mg_level_list.size() - 1)->get_coarse_op();
                int n = op->get_size()[0];
                int num_stored_elements = 0;
                if (mixed_mode == 2 || mixed_mode == 3) {
                    auto csr =
                        gko::as<gko::matrix::Csr<MixedType2, IndexType>>(op);
                    std::string file = "matrix_" +
                                       std::to_string(mg_level_list.size()) +
                                       ".mtx";
                    // output matrix
                    // write(std::ofstream(file), csr);
                    num_stored_elements = csr->get_num_stored_elements();
                } else if (mixed_mode == 1) {
                    auto csr =
                        gko::as<gko::matrix::Csr<MixedType, IndexType>>(op);
                    num_stored_elements = csr->get_num_stored_elements();
                } else {
                    auto csr = gko::as<mtx>(op);
                    num_stored_elements = csr->get_num_stored_elements();
                }
                std::cout << mg_level_list.size() << ", " << n << ", "
                          << num_stored_elements << ", " << float(n) / prev_n
                          << ", " << float(num_stored_elements) / prev_nnz
                          << ", " << float(n) / total_n << ", "
                          << float(num_stored_elements) / total_nnz
                          << std::endl;
            }
        }
    }

    int warmup = 2;
    int rep = 5;
    if (mg_mode == "preconditioner") {
        rep = 10;
    }
    std::shared_ptr<gko::LinOp> run_solver =
        (mg_mode != "cg") ? gko::as<gko::LinOp>(solver)
                          : gko::as<gko::LinOp>(cg_solver);
    auto x_run = x->clone();
    for (int i = 0; i < warmup; i++) {
        x_run->copy_from(x);
        run_solver->apply(b, x_run);
    }
    exec->synchronize();
#if ENABLE_PROFILE
    cudaProfilerStart();
#endif
    // auto prof =
    // gko::share(gko::log::ProfilerHook::create_for_executor(exec));
    // run_solver->add_logger(prof);
    // Solve system
    std::chrono::nanoseconds time(0);
    for (int i = 0; i < rep; i++) {
        x_run->copy_from(x);
        exec->synchronize();
        auto tic = std::chrono::steady_clock::now();
        run_solver->apply(b, x_run);
        exec->synchronize();
        auto toc = std::chrono::steady_clock::now();
        time += std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);
    }
// run_solver->remove_logger(prof.get());
#if ENABLE_PROFILE
    cudaProfilerStop();
#endif
    // Calculate residual
    auto res = gko::initialize<vec>({0.0}, exec);
    A->apply(one, x_run, neg_one, b);
    b->compute_norm2(res);

    std::cout << "Initial residual norm sqrt(r^T r): \n";
    write(std::cout, initres);
    std::cout << "Final residual norm sqrt(r^T r): \n";
    write(std::cout, res);

    std::string prefix = "Multigrid";
    if (mg_mode == "cg") {
        prefix = "Cg With Multigrid";
    } else if (mg_mode == "preconditioner") {
        prefix = "Multigrid preconditioner";
    }
    // Print solver statistics
    std::cout << prefix
              << " iteration count:     " << logger->get_num_iterations()
              << std::endl;
    std::cout << "Multigrid generation time [ms]: "
              << static_cast<double>(gen_time.count()) / 1000000.0 << std::endl;
    std::cout << prefix << " execution time [ms]: "
              << static_cast<double>(time.count()) / 1000000.0 / rep
              << std::endl;
    std::cout << prefix << " execution time per iteraion[ms]: "
              << static_cast<double>(time.count()) / 1000000.0 /
                     logger->get_num_iterations() / rep
              << std::endl;
}
