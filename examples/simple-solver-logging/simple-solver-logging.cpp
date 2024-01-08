// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/ginkgo.hpp>


#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>


namespace {


template <typename ValueType>
void print_vector(const std::string& name,
                  const gko::matrix::Dense<ValueType>* vec)
{
    std::cout << name << " = [" << std::endl;
    for (int i = 0; i < vec->get_size()[0]; ++i) {
        std::cout << "    " << vec->at(i, 0) << std::endl;
    }
    std::cout << "];" << std::endl;
}


}  // namespace


int main(int argc, char* argv[])
{
    // Some shortcuts
    using ValueType = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;

    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using cg = gko::solver::Cg<ValueType>;

    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor]" << std::endl;
        std::exit(-1);
    }

    // Figure out where to run the code
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

    // Read data
    auto A = share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));
    auto b = gko::read<vec>(std::ifstream("data/b.mtx"), exec);
    auto x = gko::read<vec>(std::ifstream("data/x0.mtx"), exec);

    // Let's declare a logger which prints to std::cout instead of printing to a
    // file. We log all events except for all linop factory and polymorphic
    // object events. Events masks are group of events which are provided
    // for convenience.
    std::shared_ptr<gko::log::Stream<ValueType>> stream_logger =
        gko::log::Stream<ValueType>::create(
            gko::log::Logger::all_events_mask ^
                gko::log::Logger::linop_factory_events_mask ^
                gko::log::Logger::polymorphic_object_events_mask,
            std::cout);

    // Add stream_logger to the executor
    exec->add_logger(stream_logger);

    // Add stream_logger only to the ResidualNorm criterion Factory
    // Note that the logger will get automatically propagated to every criterion
    // generated from this factory.
    const RealValueType reduction_factor{1e-7};
    using ResidualCriterionFactory =
        gko::stop::ResidualNorm<ValueType>::Factory;
    std::shared_ptr<ResidualCriterionFactory> residual_criterion =
        ResidualCriterionFactory::create()
            .with_reduction_factor(reduction_factor)
            .on(exec);
    residual_criterion->add_logger(stream_logger);

    // Generate solver
    auto solver_gen =
        cg::build()
            .with_criteria(residual_criterion,
                           gko::stop::Iteration::build().with_max_iters(20u))
            .on(exec);
    auto solver = solver_gen->generate(A);

    // First we add facilities to only print to a file. It's possible to select
    // events, using masks, e.g. only iterations mask:
    // gko::log::Logger::iteration_complete_mask. See the documentation of
    // Logger class for more information.
    std::ofstream filestream("my_file.txt");
    solver->add_logger(gko::log::Stream<ValueType>::create(
        gko::log::Logger::all_events_mask, filestream));
    solver->add_logger(stream_logger);

    // This adds a simple logger that only reports convergence state at the end
    // of the solver. Specifically it captures the last residual norm, the
    // final number of iterations, and the converged or not converged status.
    std::shared_ptr<gko::log::Convergence<ValueType>> convergence_logger =
        gko::log::Convergence<ValueType>::create();
    solver->add_logger(convergence_logger);

    // Add another logger which puts all the data in an object, we can later
    // retrieve this object in our code. Here we only have want Executor
    // and criterion check completed events.
    std::shared_ptr<gko::log::Record> record_logger =
        gko::log::Record::create(gko::log::Logger::executor_events_mask |
                                 gko::log::Logger::iteration_complete_mask);
    exec->add_logger(record_logger);
    solver->add_logger(record_logger);

    // Solve system
    solver->apply(b, x);

    // Print the residual of the last criterion check event (where
    // convergence happened)
    auto residual =
        record_logger->get().iteration_completed.back()->residual.get();
    auto residual_d = gko::as<vec>(residual);
    print_vector("Residual", residual_d);

    // Print solution
    std::cout << "Solution (x):\n";
    write(std::cout, x);
    std::cout << "Residual norm sqrt(r^T r):\n";
    write(std::cout, gko::as<vec>(convergence_logger->get_residual_norm()));
    std::cout << "Number of iterations "
              << convergence_logger->get_num_iterations() << std::endl;
    std::cout << "Convergence status " << std::boolalpha
              << convergence_logger->has_converged() << std::endl;
}
