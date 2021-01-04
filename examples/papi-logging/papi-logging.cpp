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


#include <papi.h>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <thread>


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


int init_papi_counters(std::string solver_name, std::string A_name)
{
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

    std::string simple_apply_string("sde:::ginkgo0::linop_apply_completed::");
    std::string advanced_apply_string(
        "sde:::ginkgo0::linop_advanced_apply_completed::");
    papi_add_event(simple_apply_string + solver_name, eventset);
    papi_add_event(simple_apply_string + A_name, eventset);
    papi_add_event(advanced_apply_string + A_name, eventset);

    ret_val = PAPI_start(eventset);
    if (PAPI_OK != ret_val) {
        std::cerr << "Error at PAPI_start()" << std::endl;
        std::exit(-1);
    }
    return eventset;
}


void print_papi_counters(int eventset)
{
    // Stop PAPI and read the linop_apply_completed event for all of them
    long long int values[3];
    int ret_val = PAPI_stop(eventset, values);
    if (PAPI_OK != ret_val) {
        std::cerr << "Error at PAPI_stop()" << std::endl;
        std::exit(-1);
    }

    PAPI_shutdown();

    // Print all values returned from PAPI
    std::cout << "PAPI SDE counters:" << std::endl;
    std::cout << "solver did " << values[0] << " applies." << std::endl;
    std::cout << "A did " << values[1] << " simple applies." << std::endl;
    std::cout << "A did " << values[2] << " advanced applies." << std::endl;
}


int main(int argc, char *argv[])
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

    // Read data
    auto A = share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));
    auto b = gko::read<vec>(std::ifstream("data/b.mtx"), exec);
    auto x = gko::read<vec>(std::ifstream("data/x0.mtx"), exec);

    // Generate solver
    const RealValueType reduction_factor{1e-7};
    auto solver_gen =
        cg::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(20u).on(exec),
                gko::stop::ResidualNormReduction<ValueType>::build()
                    .with_reduction_factor(reduction_factor)
                    .on(exec))
            .on(exec);
    auto solver = solver_gen->generate(A);

    // In this example, we split as much as possible the Ginkgo solver/logger
    // and the PAPI interface. Note that the PAPI ginkgo namespaces are of the
    // form sde:::ginkgo<x> where <x> starts from 0 and is incremented with
    // every new PAPI logger.
    int eventset =
        init_papi_counters(to_string(solver.get()), to_string(A.get()));


    // Create a PAPI logger and add it to relevant LinOps
    auto logger = gko::log::Papi<ValueType>::create(
        exec, gko::log::Logger::linop_apply_completed_mask |
                  gko::log::Logger::linop_advanced_apply_completed_mask);
    solver->add_logger(logger);
    A->add_logger(logger);

    // Solve system
    solver->apply(lend(b), lend(x));


    // Stop PAPI event gathering and print the counters
    print_papi_counters(eventset);

    // Print solution
    std::cout << "Solution (x): \n";
    write(std::cout, lend(x));

    // Calculate residual
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto res = gko::initialize<real_vec>({0.0}, exec);
    A->apply(lend(one), lend(x), lend(neg_one), lend(b));
    b->compute_norm2(lend(res));

    std::cout << "Residual norm sqrt(r^T r): \n";
    write(std::cout, lend(res));
}
