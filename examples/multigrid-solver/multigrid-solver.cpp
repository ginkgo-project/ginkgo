// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

#include <ginkgo/ginkgo.hpp>


using ValueType = double;
using IndexType = int;

template <typename Type>
std::shared_ptr<const gko::LinOpFactory> generate_inner(
    std::shared_ptr<gko::Executor> exec, unsigned smoother,
    double smoother_real_parameter, unsigned triangular_solver_iter)
{
    using ir = gko::solver::Ir<Type>;
    using lower_trs = gko::solver::LowerTrs<Type>;
    using ic = gko::preconditioner::Ic<lower_trs>;
    using bj = gko::preconditioner::Jacobi<Type, IndexType>;
    switch (smoother) {
    case 1:
        return gko::share(
            gko::preconditioner::Jacobi<ValueType, IndexType>::build()
                .with_max_block_size(1u)
                .on(exec));
        break;
    case 2:
        // always use double for factorization
        return gko::share(
            ic::build()
                .with_factorization(
                    gko::factorization::Ic<ValueType, int>::build().on(exec))
                .with_l_solver_factory(
                    lower_trs::build()
                        .with_algorithm(
                            gko::solver::trisolve_algorithm::syncfree)
                        .on(exec))
                .on(exec));
        break;
    case 3:
        return gko::share(
            ic::build()
                .with_factorization(
                    gko::factorization::ParIct<ValueType, int>::build()
                        .with_fill_in_limit(smoother_real_parameter)
                        .with_iterations(10u)
                        .on(exec->get_master()))
                .with_l_solver_factory(
                    lower_trs::build()
                        .with_algorithm(
                            gko::solver::trisolve_algorithm::syncfree)
                        .on(exec))
                .on(exec));
        break;
    case 4:
        return gko::share(
            gko::preconditioner::Ic<ir>::build()
                .with_factorization(
                    gko::factorization::ParIct<ValueType, int>::build()
                        .with_fill_in_limit(smoother_real_parameter)
                        .with_iterations(10u)
                        .on(exec->get_master()))
                .with_l_solver_factory(
                    ir::build()
                        .with_solver(gko::share(
                            bj::build().with_max_block_size(1u).on(exec)))
                        .with_criteria(
                            gko::stop::Iteration::build().with_max_iters(
                                triangular_solver_iter))
                        .on(exec))
                .on(exec));
        break;
    }
}


int main(int argc, char* argv[])
{
    // Some shortcuts
    using MixedType = float;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using mg = gko::solver::Multigrid;
    using pgm = gko::multigrid::Pgm<ValueType, IndexType>;

    // Print version information
    std::cout << gko::version_info::get() << std::endl;
    if (argc > 1 && argv[1] == std::string("--help")) {
        std::cout
            << "Usage:" << argv[0]
            << " executor, matrix, rhs, max_mg_levels, coarse_solver, "
               "coarsest_level_tolerance, smoother kind, smoother precision, "
               "smoother iteration, smoother parameter(fill-in), "
               "is_export, custom_prefix"
            << std::endl;
        std::exit(-1);
    }
    const std::string executor_string = argc >= 2 ? argv[1] : "reference";
    const std::string matrix_string = argc >= 3 ? argv[2] : "data/A.mtx";
    const std::string rhs_string = argc >= 4 ? argv[3] : "ones";
    const unsigned max_mg_levels =
        argc >= 5 ? static_cast<unsigned>(std::stoi(argv[4])) : 5u;
    const std::string coarse_string = argc >= 6 ? argv[5] : "direct";
    const ValueType coarsest_level_tolerance =
        argc >= 7 ? static_cast<ValueType>(std::stod(argv[6]))
                  : static_cast<ValueType>(1.0e-4);
    // smoother Jacobi(1), IC(2), ParICT(3), ParICT with IR-JACOBI (4)
    const unsigned smoother =
        argc >= 8 ? static_cast<unsigned>(std::stoi(argv[7])) : 1u;
    const std::string precision = argc >= 9 ? argv[8] : "double";
    if (precision != "double" && precision != "single" && precision != "half") {
        std::cout << "precision can only be double, single, or half. input is "
                  << precision << std::endl;
        std::exit(-1);
    }
    if (smoother == 1 && precision != "double") {
        std::cout << "Jacobi only for double" << std::endl;
        std::exit(-1);
    }
    // jacobi = 1
    // IC = 2
    // IC low = 3
    // ParICT = 4
    // ParICT low = 5
    // ParICT but using Jacobi to solve = 6
    // ParICT low but using Jacobi to solve = 7
    const unsigned smoother_iter =
        argc >= 10 ? static_cast<unsigned>(std::stoi(argv[9])) : 3u;
    const ValueType smoother_real_parameter =
        argc >= 11 ? static_cast<ValueType>(std::stod(argv[10]))
                   : static_cast<ValueType>(1.0);
    const unsigned triangular_solver_iter =
        argc >= 12 ? static_cast<unsigned>(std::stoi(argv[11])) : 10u;
    const bool export_data =
        argc >= 13 ? (argv[12] == std::string("true")) : false;
    const std::string custom_prefix = argc >= 14 ? argv[13] : "none";
    std::cout << "executor: " << executor_string << std::endl;
    std::cout << "matrix: " << matrix_string << std::endl;
    std::cout << "rhs: " << rhs_string << std::endl;
    std::cout << "max mg_levels: " << max_mg_levels << std::endl;
    std::cout << "coarse solver: " << coarse_string << std::endl;
    std::cout << "coarse tolerance: " << coarsest_level_tolerance << std::endl;
    std::cout << "smoother kind: " << smoother << std::endl;
    std::cout << "smoother precision: " << precision << std::endl;
    std::cout << "smoother iterations: " << smoother_iter << std::endl;
    std::cout
        << "smoother parameter (fill-in) or weight for Jacobi (smoother = 1): "
        << smoother_real_parameter << std::endl;
    std::cout << "triangular solver iter: " << triangular_solver_iter
              << std::endl;
    std::cout << "export intermediate data : " << export_data << std::endl;
    std::cout << "custom hierarchy prefix: " << custom_prefix << std::endl;

    // Figure out where to run the code
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [] {
                 return gko::CudaExecutor::create(
                     0, gko::ReferenceExecutor::create());
             }},
            {"hip",
             [] {
                 return gko::HipExecutor::create(
                     0, gko::ReferenceExecutor::create());
             }},
            {"dpcpp",
             [] {
                 return gko::DpcppExecutor::create(
                     0, gko::ReferenceExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string)();  // throws if not valid
    // Read data
    auto A = share(gko::read_generic<mtx>(std::ifstream(matrix_string), exec));
    // Create RHS as 1 and initial guess as 0
    gko::size_type size = A->get_size()[0];
    auto host_x = vec::create(exec->get_master(), gko::dim<2>(size, 1));
    auto host_b = vec::create(exec->get_master(), gko::dim<2>(size, 1));
    for (auto i = 0; i < size; i++) {
        host_x->at(i, 0) = 0.;
        host_b->at(i, 0) = 1.;
    }
    if (rhs_string != "ones") {
        host_b->copy_from(gko::read<vec>(std::ifstream(rhs_string), exec));
    }
    auto x = vec::create(exec);
    auto b = vec::create(exec);
    x->copy_from(host_x);
    b->copy_from(host_b);

    // Calculate initial residual by overwriting b
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto initres = gko::initialize<vec>({0.0}, exec);
    A->apply(one, x, neg_one, b);
    b->compute_norm2(initres);

    // copy b again
    b->copy_from(host_b);

    // Prepare the stopping criteria
    const gko::remove_complex<ValueType> tolerance = 1.0e-8;

    auto iter_stop = gko::share(
        gko::stop::Iteration::build().with_max_iters(100000u).on(exec));
    auto tol_stop = gko::share(gko::stop::ResidualNorm<ValueType>::build()
                                   .with_reduction_factor(tolerance)
                                   .on(exec));
    // LL' ~ A (IC)
    // smoother(b, x)
    // r = b - Ax
    // LL'y = r solve y -> L'^-1(L^-1r)
    // x += y
    // Create smoother factory (ir with ic)
    // smoother cases
    // jacobi = 1
    // IC = 2
    // IC low = 3
    // ParICT = 4
    // ParICT low = 5
    // ParICT but using Jacobi to solve = 6
    // ParICT low but using Jacobi to solve = 7
    std::shared_ptr<const gko::LinOpFactory> inner_gen;
    ValueType damping_parameter = static_cast<ValueType>(1.0);
    if (smoother == 1) {
        damping_parameter = smoother_real_parameter;
    } else {
        // other uses -1 to normalize the residual not the relaxation factor
        damping_parameter = -gko::one<ValueType>();
    }
    if (precision == "double") {
        inner_gen = generate_inner<double>(
            exec, smoother, smoother_real_parameter, triangular_solver_iter);
    } else if (precision == "single") {
        inner_gen = generate_inner<float>(
            exec, smoother, smoother_real_parameter, triangular_solver_iter);
    } else if (precision == "half") {
        inner_gen = generate_inner<gko::half>(
            exec, smoother, smoother_real_parameter, triangular_solver_iter);
    }

    // Create RestrictProlong factory
    auto smoother_gen = gko::share(gko::solver::build_smoother(
        inner_gen, smoother_iter, damping_parameter));
    std::vector<std::shared_ptr<const gko::LinOpFactory>> mg_level_gen;
    if (custom_prefix == std::string("none")) {
        auto mg_level =
            gko::share(pgm::build().with_deterministic(true).on(exec));
        mg_level_gen.emplace_back(mg_level);
    } else {
        for (unsigned int i = 0; i < max_mg_levels; i++) {
            auto coarse = share(gko::read_generic<mtx>(
                std::ifstream(custom_prefix + "_" + std::to_string(i + 1) +
                              ".mtx"),
                exec));
            auto rst = share(gko::read_generic<mtx>(
                std::ifstream(custom_prefix + "_r_" + std::to_string(i) +
                              ".mtx"),
                exec));
            auto prl = share(rst->conj_transpose());
            auto mg_level =
                share(gko::multigrid::CustomCoarse<ValueType>::build()
                          .with_coarse(coarse)
                          .with_restriction(rst)
                          .with_prologation(prl)
                          .on(exec));
            mg_level_gen.emplace_back(mg_level);
        }
    }
    // Create CoarsesSolver factory
    std::shared_ptr<gko::LinOpFactory> coarsest_solver_gen = nullptr;
    if (coarse_string == std::string("direct")) {
        coarsest_solver_gen = gko::share(
            gko::experimental::solver::Direct<ValueType, IndexType>::build()
                .with_factorization(gko::experimental::factorization::Cholesky<
                                    ValueType, IndexType>::build())
                .on(exec));
    } else {
        // cg
        coarsest_solver_gen = gko::share(
            gko::solver::Cg<ValueType>::build()
                .with_criteria(
                    gko::stop::ResidualNorm<>::build().with_reduction_factor(
                        coarsest_level_tolerance))
                .on(exec));
    }
    // Create multigrid factory
    auto multigrid_gen =
        gko::share(mg::build()
                       .with_max_levels(max_mg_levels)
                       .with_min_coarse_rows(2u)
                       .with_pre_smoother(smoother_gen)
                       .with_post_uses_pre(true)
                       .with_mg_level(mg_level_gen)
                       .with_coarsest_solver(coarsest_solver_gen)
                       .with_criteria(iter_stop, tol_stop)
                       .on(exec));

    std::chrono::nanoseconds gen_time(0);
    auto gen_tic = std::chrono::steady_clock::now();
    // auto solver = solver_gen->generate(A);
    auto solver = multigrid_gen->generate(A);
    exec->synchronize();
    auto gen_toc = std::chrono::steady_clock::now();
    gen_time +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(gen_toc - gen_tic);

    // Add logger
    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create();
    solver->add_logger(logger);

    {
        // information
        std::cout << "multigrid information: " << std::endl;
        auto mg_level_list = solver->get_mg_level_list();
        std::cout << "Level: " << mg_level_list.size() << std::endl;
        int total_n = solver->get_system_matrix()->get_size()[0];
        int total_nnz = gko::as<mtx>(solver->get_system_matrix())
                            ->get_num_stored_elements();
        int prev_n = total_n;
        int prev_nnz = total_nnz;
        std::cout << "0, " << prev_n << ", " << prev_nnz
                  << ", prev_n(%), prev_nnz(%), total_n(%), total_nnz(%)"
                  << std::endl;
        for (int i = 1; i < mg_level_list.size(); i++) {
            auto op = mg_level_list.at(i)->get_fine_op();
            int n = op->get_size()[0];
            int num_stored_elements = 0;
            auto csr = gko::as<mtx>(op);
            if (export_data) {
                std::string filename =
                    "data/A_mg_" + std::to_string(i) + ".mtx";
                std::ofstream ofs(filename);
                gko::write(ofs, csr);
            }
            num_stored_elements = csr->get_num_stored_elements();
            std::cout << i << ", " << n << ", " << num_stored_elements << ", "
                      << float(n) / prev_n << ", "
                      << float(num_stored_elements) / prev_nnz << ", "
                      << float(n) / total_n << ", "
                      << float(num_stored_elements) / total_nnz << std::endl;
            prev_n = n;
            prev_nnz = num_stored_elements;
        }
        {
            // for coarse matrix
            auto op =
                mg_level_list.at(mg_level_list.size() - 1)->get_coarse_op();
            auto csr = gko::as<mtx>(op);
            int n = op->get_size()[0];
            int num_stored_elements = csr->get_num_stored_elements();
            std::cout << mg_level_list.size() << ", " << n << ", "
                      << num_stored_elements << ", " << float(n) / prev_n
                      << ", " << float(num_stored_elements) / prev_nnz << ", "
                      << float(n) / total_n << ", "
                      << float(num_stored_elements) / total_nnz << std::endl;
            if (export_data) {
                std::string filename = "data/A_mg_" +
                                       std::to_string(mg_level_list.size()) +
                                       ".mtx";
                std::ofstream ofs(filename);
                ofs << std::setprecision(std::numeric_limits<double>::digits10 +
                                         1);
                gko::write(ofs, csr);
            }
        }
        if (export_data) {
            // for smoother
            auto presmoother_list = solver->get_pre_smoother_list();
            std::cout << "extract presmooth list: " << presmoother_list.size()
                      << std::endl;
            for (int i = 0; i < presmoother_list.size(); i++) {
                auto op =
                    gko::as<gko::solver::Ir<ValueType>>(presmoother_list.at(i));
                auto l_matrix = gko::as<mtx>(
                    gko::as<gko::solver::LowerTrs<ValueType>>(
                        gko::as<gko::preconditioner::Ic<
                            gko::solver::LowerTrs<ValueType>>>(op->get_solver())
                            ->get_l_solver())
                        ->get_system_matrix());
                std::string filename = "data/A_l_" + std::to_string(i) + ".mtx";
                std::ofstream ofs(filename);
                ofs << std::setprecision(std::numeric_limits<double>::digits10 +
                                         1);
                gko::write(ofs, l_matrix);
            }
            {
                // for Restrict/Prolong
                auto mg_level_list = solver->get_mg_level_list();
                for (int i = 0; i < mg_level_list.size(); i++) {
                    auto op =
                        gko::as<gko::matrix::SparsityCsr<ValueType, IndexType>>(
                            mg_level_list.at(i)->get_restrict_op());
                    if (export_data) {
                        std::string filename =
                            "data/A_mg_r_" + std::to_string(i) + ".mtx";
                        std::ofstream ofs(filename);
                        ofs << std::setprecision(
                            std::numeric_limits<double>::digits10 + 1);
                        gko::write(ofs, op);
                    }
                }
            }
        }
    }


    // Solve system
    exec->synchronize();
    std::chrono::nanoseconds time(0);
    auto tic = std::chrono::steady_clock::now();
    solver->apply(b, x);
    exec->synchronize();
    auto toc = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);

    // Calculate residual explicitly, because the residual is not
    // available inside of the multigrid solver
    auto res = gko::initialize<vec>({0.0}, exec);
    A->apply(one, x, neg_one, b);
    b->compute_norm2(res);

    std::cout << "Initial residual norm sqrt(r^T r): \n";
    write(std::cout, initres);
    std::cout << "Final residual norm sqrt(r^T r): \n";
    write(std::cout, res);

    // Print solver statistics
    std::cout << "Multigrid iteration count:     "
              << logger->get_num_iterations() << std::endl;
    std::cout << "Multigrid generation time [ms]: "
              << static_cast<double>(gen_time.count()) / 1000000.0 << std::endl;
    std::cout << "Multigrid execution time [ms]: "
              << static_cast<double>(time.count()) / 1000000.0 << std::endl;
    std::cout << "Multigrid execution time per iteration[ms]: "
              << static_cast<double>(time.count()) / 1000000.0 /
                     logger->get_num_iterations()
              << std::endl;
}
