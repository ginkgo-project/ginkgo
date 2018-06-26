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

c++ -O3 -std=c++11 -o ginkgo_overhead ginkgo_overhead.cpp -I../.. \
    -L. -lginkgo -lginkgo_reference -lginkgo_omp -lginkgo_cuda

(if ginkgo was built in debug mode, append 'd' to every library name)

Now you should be able to run the program using:

env LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} ./ginkgo_overhead

*****************************<COMPILATION>**********************************/


#include <include/ginkgo.hpp>


#include <chrono>
#include <cmath>
#include <iostream>


[[noreturn]] void print_usage_and_exit(const char *name) {
    std::cerr << "Usage: " << name << " [NUM_ITERS]" << std::endl;
    std::exit(-1);
}


int main(int argc, char *argv[])
{
    using vec = gko::matrix::Dense<>;
    using mtx = gko::matrix::Dense<>;
    using cg = gko::solver::Cg<>;

    long num_iters = 1000000;
    if (argc > 2) {
        print_usage_and_exit(argv[0]);
    }
    if (argc == 2) {
        num_iters = std::atol(argv[1]);
        if (num_iters == 0) {
            print_usage_and_exit(argv[0]);
        }
    }

    std::cout << gko::version_info::get() << std::endl;

    auto exec = gko::ReferenceExecutor::create();

    auto cg_factory =
        cg::Factory::create().with_max_iters(num_iters).on_executor(exec);
    auto A = gko::initialize<mtx>({1.0}, exec);
    auto b = gko::initialize<vec>({std::nan("")}, exec);
    auto x = gko::initialize<vec>({0.0}, exec);

    auto tic = std::chrono::system_clock::now();

    auto solver = cg_factory->generate(gko::give(A));
    solver->apply(gko::lend(x), gko::lend(b));
    exec->synchronize();

    auto tac = std::chrono::system_clock::now();

    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(tac - tic);
    std::cout << "Running " << num_iters
              << " iterations of the CG solver took a total of "
              << 1.0 * time.count() / std::nano::den << " seconds." << std::endl
              << "\tAverage library overhead:     "
              << 1.0 * time.count() / num_iters << " [nanoseconds / iteration]"
              << std::endl;
}
