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
    + gpu/

to this directory.

Then compile the file with the following command line:

c++ -std=c++11 -o asynchronous_stopping_criterion  \
    asynchronous_stopping_criterion.cpp byinteration.cpp -I../.. \
    -L. -lginkgo -lginkgo_reference -lginkgo_omp -lginkgo_gpu

(if ginkgo was built in debug mode, append 'd' to every library name)

Now you should be able to run the program using:

env LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} ./asynchronous_stopping_criterion

*****************************<COMPILATION>**********************************/

#include <include/ginkgo.hpp>
#include "byinteraction.hpp"


#include <iostream>
#include <string>
#include <thread>


void run_solver(volatile bool *stop_iteration_process,
                std::shared_ptr<gko::Executor> exec)
{
    using mtx = gko::matrix::Csr<>;
    using vec = gko::matrix::Dense<>;
    using bicg = gko::solver::Bicgstab<>;

    // Read Data
    auto A = gko::share(gko::read<mtx>("data/A.mtx", exec));
    auto b = gko::read<vec>("data/b.mtx", exec);
    auto x = gko::read<vec>("data/x0.mtx", exec);

    // Create solver factory and solve system
    bicg::Factory::create()
        .with_criterion(gko::stop::ByInteraction::Factory::create()
                            .with_stop_iteration_process(stop_iteration_process)
                            .on_executor(exec))
        .on_executor(exec)
        ->generate(A)
        ->apply(gko::lend(b), gko::lend(x));

    std::cout << "Solver stopped" << std::endl;


    // Print result
    auto h_x = gko::clone(exec->get_master(), x);
    std::cout << "x = [" << std::endl;
    for (int i = 0; i < h_x->get_size().num_rows; ++i) {
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


int main(int argc, char *argv[])
{
    // Some shortcuts

    // Figure out where to run the code
    std::shared_ptr<gko::Executor> exec;
    if (argc == 1 || std::string(argv[1]) == "reference") {
        exec = gko::ReferenceExecutor::create();
    } else if (argc == 2 && std::string(argv[1]) == "omp") {
        exec = gko::OmpExecutor::create();
    } else if (argc == 2 && std::string(argv[1]) == "gpu" &&
               gko::GpuExecutor::get_num_devices() > 0) {
        exec = gko::GpuExecutor::create(0, gko::OmpExecutor::create());
    } else {
        std::cerr << "Usage: " << argv[0] << " [executor]" << std::endl;
        std::exit(-1);
    }

    // Declare a user controled boolean for the iteration process
    volatile bool stop_iteration_process{};

    // Create a new a thread to launch the solver
    std::thread t(run_solver, &stop_iteration_process, exec);

    // Look for an input command "stop" in the console, which sets the boolean
    // to true
    std::string command;
    while (std::cin >> command) {
        if (command == "stop") {
            break;
        } else {
            std::cout << "Unknown command" << std::endl;
        }
    }
    std::cout << "User input command 'stop' - The solver will stop!"
              << std::endl;
    stop_iteration_process = true;
    t.join();
}
