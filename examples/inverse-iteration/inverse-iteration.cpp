// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/ginkgo.hpp>


#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>


int main(int argc, char* argv[])
{
    // Some shortcuts
    using precision = std::complex<double>;
    using real_precision = gko::remove_complex<precision>;
    using vec = gko::matrix::Dense<precision>;
    using real_vec = gko::matrix::Dense<real_precision>;
    using mtx = gko::matrix::Csr<precision>;
    using solver_type = gko::solver::Bicgstab<precision>;

    using std::abs;
    using std::sqrt;

    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    std::cout << std::scientific << std::setprecision(8) << std::showpos;

    // Figure out where to run the code
    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor]" << std::endl;
        std::exit(-1);
    }

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [] {
                 return gko::CudaExecutor::create(0,
                                                  gko::OmpExecutor::create());
             }},
            {"hip",
             [] {
                 return gko::HipExecutor::create(0, gko::OmpExecutor::create());
             }},
            {"dpcpp",
             [] {
                 return gko::DpcppExecutor::create(0,
                                                   gko::OmpExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string)();  // throws if not valid

    auto this_exec = exec->get_master();

    // linear system solver parameters
    auto system_max_iterations = 100u;
    auto system_residual_goal = real_precision{1e-16};

    // eigensolver parameters
    auto max_iterations = 20u;
    auto residual_goal = real_precision{1e-8};
    auto z = precision{20.0, 2.0};

    // Read data
    auto A = share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));

    // Generate shifted matrix  A - zI
    // - we avoid duplicating memory by not storing both A and A - zI, but
    //   compute A - zI on the fly by using Ginkgo's utilities for creating
    //   linear combinations of operators
    auto one = share(gko::initialize<vec>({precision{1.0}}, exec));
    auto neg_one = share(gko::initialize<vec>({-precision{1.0}}, exec));
    auto neg_z = gko::initialize<vec>({-z}, exec);

    auto system_matrix = share(gko::Combination<precision>::create(
        one, A, gko::initialize<vec>({-z}, exec),
        gko::matrix::Identity<precision>::create(exec, A->get_size()[0])));

    // Generate solver operator  (A - zI)^-1
    auto solver =
        solver_type::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(
                               system_max_iterations),
                           gko::stop::ResidualNorm<precision>::build()
                               .with_reduction_factor(system_residual_goal))
            .on(exec)
            ->generate(system_matrix);

    // inverse iterations

    // start with guess [1, 1, ..., 1]
    auto x = [&] {
        auto work = vec::create(this_exec, gko::dim<2>{A->get_size()[0], 1});
        const auto n = work->get_size()[0];
        for (int i = 0; i < n; ++i) {
            work->get_values()[i] = precision{1.0} / sqrt(n);
        }
        return clone(exec, work);
    }();
    auto y = clone(x);
    auto tmp = clone(x);
    auto norm = gko::initialize<real_vec>({1.0}, exec);
    auto inv_norm = clone(this_exec, one);
    auto g = clone(one);

    for (auto i = 0u; i < max_iterations; ++i) {
        std::cout << "{ ";
        // (A - zI)y = x
        solver->apply(x, y);
        system_matrix->apply(one, y, neg_one, x);
        x->compute_norm2(norm);
        std::cout << "\"system_residual\": "
                  << clone(this_exec, norm)->get_values()[0] << ", ";
        x->copy_from(y);
        // x = y / || y ||
        x->compute_norm2(norm);
        inv_norm->get_values()[0] =
            real_precision{1.0} / clone(this_exec, norm)->get_values()[0];
        x->scale(clone(exec, inv_norm));
        // g = x^* A x
        A->apply(x, tmp);
        x->compute_dot(tmp, g);
        auto g_val = clone(this_exec, g)->get_values()[0];
        std::cout << "\"eigenvalue\": " << g_val << ", ";
        // ||Ax - gx|| < tol * g
        auto v = gko::initialize<vec>({-g_val}, exec);
        tmp->add_scaled(v, x);
        tmp->compute_norm2(norm);
        auto res_val = clone(exec->get_master(), norm)->get_values()[0];
        std::cout << "\"residual\": " << res_val / g_val << " }," << std::endl;
        if (abs(res_val) < residual_goal * abs(g_val)) {
            break;
        }
    }
}
