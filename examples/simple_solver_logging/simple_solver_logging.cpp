/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

/*****************************<COMPILATION>***********************************
The easiest way to build the example solver is to use the script provided:
./build.sh <PATH_TO_GINKGO_BUILD_DIR>

Ginkgo should be compiled with `-DBUILD_REFERENCE=on` option.

Alternatively, you can setup the configuration manually:

Go to the <PATH_TO_GINKGO_BUILD_DIR> directory and copy the shared
libraries located in the following subdirectories:

    + core/
    + core/device_hooks/
    + reference/
    + omp/
    + cuda/

to this directory.

Then compile the file with the following command line:

c++ -std=c++11 -o simple_solver_logging simple_solver_logging.cpp \
   -I../.. -L. -lginkgo -lginkgo_reference -lginkgo_omp -lginkgo_cuda

(if ginkgo was built in debug mode, append 'd' to every library name)

Now you should be able to run the program using:

env LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} ./simple_solver_logging

*****************************<COMPILATION>**********************************/

#include <include/ginkgo.hpp>
#include <iomanip>
#include <iostream>
#include <string>

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
    } else {
        std::cerr << "Usage: " << argv[0] << " [executor]" << std::endl;
        std::exit(-1);
    }

    // Read data
    auto A = gko::share(gko::read<mtx>("data/A.mtx", exec));
    auto b = gko::read<vec>("data/b.mtx", exec);
    auto x = gko::read<vec>("data/x0.mtx", exec);

    // Generate solver
    auto solver_gen =
        cg::Factory::create()
            .with_criterion(
                gko::stop::Combined::Factory::create()
                    .with_criteria(
                        gko::stop::Iteration::Factory::create()
                            .with_max_iters(20u)
                            .on_executor(exec),
                        gko::stop::ResidualNormReduction<>::Factory::create()
                            .with_reduction_factor(1e-20)
                            .on_executor(exec))
                    .on_executor(exec))
            .on_executor(exec);
    auto solver = solver_gen->generate(A);

    // Here begins the actual logging example:
    // First we add facilities to only print to a file. It's possible to select
    // events, using masks, e.g. only iterations mask:
    // gko::log::Logger::iteration_complete_mask. See the documentation of
    // Logger class for more information.
    std::ofstream filestream("my_file.txt");
    solver->add_logger(gko::log::Stream<>::create(
        exec, gko::log::Logger::all_events_mask, filestream));

    // Another logger which prints to std::cout instead of printing to a file
    // We log all events except for apply.
    solver->add_logger(gko::log::Stream<>::create(
        exec, gko::log::Logger::all_events_mask ^ gko::log::Logger::apply_mask,
        std::cout));

    // Add another logger which puts all the data in an object, we can later
    // retrieve this object in our code
    std::shared_ptr<gko::log::Record> record_logger =
        gko::log::Record::create(exec, gko::log::Logger::all_events_mask);
    solver->add_logger(record_logger);

    // Solve system
    solver->apply(gko::lend(b), gko::lend(x));

    // Finally, get the data from `record_logger` and print an element
    auto residual = record_logger->get().residuals.back().get();
    auto residual_d = gko::as<gko::matrix::Dense<>>(residual);
    // Set precision as needed for output
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "Residual(1) : " << residual_d->at(1) << std::endl;

    // Print result
    auto h_x = gko::clone(exec->get_master(), x);
    std::cout << "x = [" << std::endl;
    for (int i = 0; i < h_x->get_size()[0]; ++i) {
        std::cout << "    " << h_x->at(i, 0) << std::endl;
    }
    std::cout << "];" << std::endl;

    // Calculate residual
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto res = gko::initialize<vec>({0.0}, exec);
    A->apply(gko::lend(one), gko::lend(x), gko::lend(neg_one), gko::lend(b));
    b->compute_dot(gko::lend(b), gko::lend(res));

    auto h_res = gko::clone(exec->get_master(), res);
    std::cout << "res = " << std::sqrt(h_res->at(0, 0)) << ";" << std::endl;
}
