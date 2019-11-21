/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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
#include <iomanip>
#include <iostream>
#include <string>


namespace {


void print_vector(const std::string &name, const gko::matrix::Dense<> *vec)
{
    std::cout << name << " = [" << std::endl;
    for (int i = 0; i < vec->get_size()[0]; ++i) {
        std::cout << "    " << vec->at(i, 0) << std::endl;
    }
    std::cout << "];" << std::endl;
}


}  // namespace


int main(int argc, char *argv[])
{
    // Some shortcuts
    using vec = gko::matrix::Dense<>;
    using mtx = gko::matrix::Csr<>;
    using cg = gko::solver::Cg<>;

    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    // Figure out where to run the code
    std::shared_ptr<gko::Executor> exec;
    if (argc == 1 || std::string(argv[1]) == "reference") {
        exec = gko::ReferenceExecutor::create();
    } else if (argc == 2 && std::string(argv[1]) == "omp") {
        exec = gko::OmpExecutor::create();
    } else if (argc == 2 && std::string(argv[1]) == "cuda" &&
               gko::CudaExecutor::get_num_devices() > 0) {
        exec = gko::CudaExecutor::create(0, gko::OmpExecutor::create());
    } else if (argc == 2 && std::string(argv[1]) == "hip" &&
               gko::HipExecutor::get_num_devices() > 0) {
        exec = gko::HipExecutor::create(0, gko::OmpExecutor::create());
    } else {
        std::cerr << "Usage: " << argv[0] << " [executor]" << std::endl;
        std::exit(-1);
    }

    // Read data
    auto A = share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));
    auto b = gko::read<vec>(std::ifstream("data/b.mtx"), exec);
    auto x = gko::read<vec>(std::ifstream("data/x0.mtx"), exec);

    // Let's declare a logger which prints to std::cout instead of printing to a
    // file. We log all events except for all linop factory and polymorphic
    // object events. Events masks are group of events which are provided
    // for convenience.
    std::shared_ptr<gko::log::Stream<>> stream_logger =
        gko::log::Stream<>::create(
            exec, exec->get_mem_space(),
            gko::log::Logger::all_events_mask ^
                gko::log::Logger::linop_factory_events_mask ^
                gko::log::Logger::polymorphic_object_events_mask,
            std::cout);

    // Add stream_logger to the executor
    exec->add_logger(stream_logger);

    // Add stream_logger only to the ResidualNormReduction criterion Factory
    // Note that the logger will get automatically propagated to every criterion
    // generated from this factory.
    using ResidualCriterionFactory =
        gko::stop::ResidualNormReduction<>::Factory;
    std::shared_ptr<ResidualCriterionFactory> residual_criterion =
        ResidualCriterionFactory::create().with_reduction_factor(1e-20).on(
            exec);
    residual_criterion->add_logger(stream_logger);

    // Generate solver
    auto solver_gen =
        cg::build()
            .with_criteria(
                residual_criterion,
                gko::stop::Iteration::build().with_max_iters(20u).on(exec))
            .on(exec);
    auto solver = solver_gen->generate(A);


    // First we add facilities to only print to a file. It's possible to select
    // events, using masks, e.g. only iterations mask:
    // gko::log::Logger::iteration_complete_mask. See the documentation of
    // Logger class for more information.
    std::ofstream filestream("my_file.txt");
    solver->add_logger(gko::log::Stream<>::create(
        exec, exec->get_mem_space(), gko::log::Logger::all_events_mask,
        filestream));
    solver->add_logger(stream_logger);

    // Add another logger which puts all the data in an object, we can later
    // retrieve this object in our code. Here we only have want Executor
    // and criterion check completed events.
    std::shared_ptr<gko::log::Record> record_logger = gko::log::Record::create(
        exec, exec->get_mem_space(),
        gko::log::Logger::executor_events_mask |
            gko::log::Logger::criterion_check_completed_mask);
    exec->add_logger(record_logger);
    residual_criterion->add_logger(record_logger);

    // Solve system
    solver->apply(lend(b), lend(x));

    // Finally, get some data from `record_logger` and print the last memory
    // location copied
    auto &last_copy = record_logger->get().copy_completed.back();
    std::cout << "Last memory copied was of size " << std::hex
              << std::get<0>(*last_copy).num_bytes << " FROM Memory Space"
              << std::get<0>(*last_copy).mem_space << " pointer "
              << std::get<0>(*last_copy).location << " TO Memory Space "
              << std::get<1>(*last_copy).mem_space << " pointer "
              << std::get<1>(*last_copy).location << std::dec << std::endl;
    // Also print the residual of the last criterion check event (where
    // convergence happened)
    auto residual =
        record_logger->get().criterion_check_completed.back()->residual.get();
    auto residual_d = gko::as<gko::matrix::Dense<>>(residual);
    print_vector("Residual", residual_d);

    // Print solution
    std::cout << "Solution (x): \n";
    write(std::cout, lend(x));

    // Calculate residual
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto res = gko::initialize<vec>({0.0}, exec);
    A->apply(lend(one), lend(x), lend(neg_one), lend(b));
    b->compute_norm2(lend(res));

    std::cout << "Residual norm sqrt(r^T r): \n";
    write(std::cout, lend(res));
}
