// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <complex>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include <ginkgo/ginkgo.hpp>


int main(int argc, char* argv[])
{
    // Some shortcuts
    using ValueType = std::complex<double>;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;

    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using gmres = gko::solver::Gmres<ValueType>;

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

    // Read data
    auto A = gko::share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));
    auto b = gko::read<vec>(std::ifstream("data/b.mtx"), exec);
    auto x = gko::read<vec>(std::ifstream("data/x0.mtx"), exec);

    // Generate incomplete factors using ParILU
    auto cuda_ilu_fact =
        gko::factorization::Ilu<ValueType, IndexType>::build().on(exec);
    // Generate concrete factorization for input matrix
    auto cuda_ilu = gko::share(cuda_ilu_fact->generate(A));
    std::cout << "Finish CUDA ILU" << std::endl;
    {
        std::ofstream L("cL.mtx");
        std::ofstream U("cU.mtx");
        gko::write(L, cuda_ilu->get_l_factor());
        gko::write(U, cuda_ilu->get_u_factor());
    }
    // Generate an ILU preconditioner factory by setting lower and upper
    // triangular solver - in this case the exact triangular solves
    auto ilu_pre_factory =
        gko::preconditioner::Ilu<gko::solver::LowerTrs<ValueType, IndexType>,
                                 gko::solver::UpperTrs<ValueType, IndexType>,
                                 false>::build()
            .on(exec);

    auto gko_ilu = gko::share(
        gko::experimental::factorization::Lu<ValueType, IndexType>::build()
            .with_symbolic_algorithm(
                gko::experimental::factorization::symbolic_type::incomplete)
            .on(exec)
            ->generate(A));
    std::cout << "Finish GKO ILU" << std::endl;
    {
        auto unpack = gko_ilu->unpack();
        std::ofstream L("gL.mtx");
        std::ofstream U("gU.mtx");
        gko::write(L, unpack->get_lower_factor());
        gko::write(U, unpack->get_upper_factor());
    }

    std::exit(-1);

    // Use incomplete factors to generate ILU preconditioner
    auto ilu_preconditioner = gko::share(ilu_pre_factory->generate(gko_ilu));


    // Use preconditioner inside GMRES solver factory
    // Generating a solver factory tied to a specific preconditioner makes sense
    // if there are several very similar systems to solve, and the same
    // solver+preconditioner combination is expected to be effective.
    const RealValueType reduction_factor{1e-7};
    auto ilu_gmres_factory =
        gmres::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1000u),
                           gko::stop::ResidualNorm<ValueType>::build()
                               .with_reduction_factor(reduction_factor))
            .with_generated_preconditioner(ilu_preconditioner)
            .on(exec);

    // Generate preconditioned solver for a specific target system
    auto ilu_gmres = ilu_gmres_factory->generate(A);

    // Solve system
    ilu_gmres->apply(b, x);

    // Print solution
    std::cout << "Solution (x):\n";
    write(std::cout, x);

    // Calculate residual
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto res = gko::initialize<real_vec>({0.0}, exec);
    A->apply(one, x, neg_one, b);
    b->compute_norm2(res);

    std::cout << "Residual norm sqrt(r^T r):\n";
    write(std::cout, res);
}
