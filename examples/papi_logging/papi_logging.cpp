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

c++ -std=c++11 -o simple_solver simple_solver.cpp -I../.. \
    -L. -lginkgo -lginkgo_reference -lginkgo_omp -lginkgo_cuda

(if ginkgo was built in debug mode, append 'd' to every library name)

Now you should be able to run the program using:

env LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} ./simple_solver

*****************************<COMPILATION>**********************************/

#include <include/ginkgo.hpp>


#include <papi.h>
#include <fstream>
#include <iostream>
#include <string>


namespace {


void papi_add_event(const std::string &event_name, int &eventset)
{
    int code;
    int ret_val = PAPI_event_name_to_code(event_name.c_str(), &code);
    if (PAPI_OK != ret_val) {
        std::cerr << "Error at PAPI_name_to_code()" << std::endl;
        std::exit(-1);
    }

    ret_val = PAPI_add_event(eventset, code);
    if (PAPI_OK != ret_val) {
        std::cerr << "Error at PAPI_name_to_code()" << std::endl;
        std::exit(-1);
    }
}


template <typename T>
std::string to_string(T *ptr)
{
    std::ostringstream os;
    os << reinterpret_cast<gko::uintptr>(ptr);
    return os.str();
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
    } else {
        std::cerr << "Usage: " << argv[0] << " [executor]" << std::endl;
        std::exit(-1);
    }

    // Read data
    auto A = share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));
    auto b = gko::read<vec>(std::ifstream("data/b.mtx"), exec);
    auto x = gko::read<vec>(std::ifstream("data/x0.mtx"), exec);

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

    // This is the start of the interesting part. This example is the same as
    // simple_solver except for the use of the PAPI logger.
    // Create a PAPI logger and add it to relevant LinOps
    auto logger = gko::log::Papi<>::create(
        exec, gko::log::Logger::linop_apply_completed_mask |
                  gko::log::Logger::linop_advanced_apply_completed_mask);
    solver->add_logger(logger);
    A->add_logger(logger);

    // Initialize PAPI, add events and start it up
    int eventset = PAPI_NULL;
    int ret_val = PAPI_library_init(PAPI_VER_CURRENT);
    if (ret_val != PAPI_VER_CURRENT) {
        std::cerr << "Error at PAPI_library_init()" << std::endl;
        std::exit(-1);
    }
    ret_val = PAPI_create_eventset(&eventset);
    if (PAPI_OK != ret_val) {
        std::cerr << "Error at PAPI_create_eventset()" << std::endl;
        std::exit(-1);
    }

    std::string simple_apply_string("sde:::" + logger->get_handle_name() +
                                    "::linop_apply_completed::");
    std::string advanced_apply_string("sde:::" + logger->get_handle_name() +
                                      "::linop_advanced_apply_completed::");
    papi_add_event(simple_apply_string + to_string(solver.get()), eventset);
    papi_add_event(simple_apply_string + to_string(A.get()), eventset);
    papi_add_event(advanced_apply_string + to_string(A.get()), eventset);

    ret_val = PAPI_start(eventset);
    if (PAPI_OK != ret_val) {
        std::cerr << "Error at PAPI_start()" << std::endl;
        std::exit(-1);
    }

    // Solve system
    solver->apply(lend(b), lend(x));

    // Stop PAPI and read the linop_apply_completed event for all of them
    long long int values[3];
    ret_val = PAPI_stop(eventset, values);
    if (PAPI_OK != ret_val) {
        std::cerr << "Error at PAPI_stop()" << std::endl;
        std::exit(-1);
    }

    PAPI_shutdown();

    // Print all values returned from PAPI
    std::cout << "PAPI SDE counter:" << std::endl;
    std::cout << "solver did " << values[0] << " applies." << std::endl;
    std::cout << "A did " << values[1] << " simple applies." << std::endl;
    std::cout << "A did " << values[2] << " advanced applies." << std::endl;

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
