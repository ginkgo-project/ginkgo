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


#include <ginkgo/ginkgo.hpp>


#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>


int main(int argc, char* argv[])
{
    // Some shortcuts
    using ValueType = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;

    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using lower_trs = gko::solver::LowerTrs<ValueType, IndexType>;
    using upper_trs = gko::solver::UpperTrs<ValueType, IndexType>;

    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor]" << std::endl;
        std::exit(-1);
    }

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
                 return gko::DpcppExecutor::create(0,
                                                   gko::OmpExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string)();  // throws if not valid

    auto mat_string = argc >= 3 ? argv[2] : "data/A.mtx";
    // Read data
    auto A = gko::share(gko::read<mtx>(std::ifstream(mat_string), exec));

    // Create MC64 and AMD reordering and scaling
    auto preprocessing_fact =
        gko::share(gko::reorder::Mc64<ValueType, IndexType>::build().on(exec));
    auto preprocessing = gko::share(preprocessing_fact->generate(A));

    // Create reusable GLU factory for first matrix
    auto lu_fact = gko::share(
        gko::factorization::Glu<ValueType, IndexType>::build_reusable().on(
            exec, A.get(), preprocessing.get()));
    auto inner_solver_fact = gko::share(gko::preconditioner::Ilu<>::build()
                                            .with_factorization_factory(lu_fact)
                                            .on(exec));
    auto solver_fact = gko::share(
        gko::solver::Gmres<>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(20u).on(exec),
                gko::stop::ResidualNorm<>::build()
                    .with_baseline(gko::stop::mode::absolute)
                    .with_reduction_factor(1e-8)
                    .on(exec))
            .with_krylov_dim(5u)
            .with_preconditioner(inner_solver_fact)
            .on(exec));

    auto reordered_solver_fact = gko::solver::ScaledReordered<>::build()
                                     .with_solver(solver_fact)
                                     .with_reordering(preprocessing)
                                     .on(exec);

    auto n = A->get_size()[0];
    auto host_b = vec::create(exec->get_master(), gko::dim<2>{n, 1});
    auto host_x = vec::create(exec->get_master(), gko::dim<2>{n, 1});
    for (auto i = 0; i < n; i++) {
        host_b->at(i, 0) = 1.;
        host_x->at(i, 0) = 0.;
    }
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto res = gko::initialize<real_vec>({1e10}, exec->get_master());

    for (auto i = 0; i <= argc - 3; i++) {
        auto x = vec::create(exec);
        x->copy_from(host_b.get());
        auto b = vec::create_with_config_of(x.get());
        A->apply(x.get(), b.get());
        x->copy_from(host_x.get());

        auto solver = reordered_solver_fact->generate(A);

        solver->apply(b.get(), x.get());

        // Print solution
        // std::ofstream x_out{"x.mtx"};
        // write(x_out, gko::lend(x));

        x->add_scaled(neg_one.get(), host_b.get());
        x->compute_norm2(gko::lend(res));
        std::cout << "Final error norm sqrt(r^T r):\n";
        write(std::cout, gko::lend(res));

        if (3 + i == argc) break;
        mat_string = argv[3 + i];
        A = gko::share(gko::read<mtx>(std::ifstream(mat_string), exec));
    }
}
