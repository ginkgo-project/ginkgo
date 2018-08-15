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

#include <include/ginkgo.hpp>


#include <functional>
#include <iostream>
#include <map>


#include <gflags/gflags.h>


// Command-line arguments
DEFINE_uint32(device_id, 0, "ID of the device where to run the code");

DEFINE_uint32(max_iters, 1000,
              "Maximal number of iterations the solver will be run for");

DEFINE_double(rel_res_goal, 1e-6, "The relative residual goal of the solver");

const std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
    executor_factory{
        {"reference", [] { return gko::ReferenceExecutor::create(); }},
        {"omp", [] { return gko::OmpExecutor::create(); }},
        {"cuda", [] {
             return gko::CudaExecutor::create(FLAGS_device_id,
                                              gko::OmpExecutor::create());
         }}};

DEFINE_string(
    executor, "reference",
    "The executor used to run the solver, one of: reference, omp, cuda");
bool validate_executor(const char *flag_name, const std::string &value)
{
    if (executor_factory.count(value) == 0) {
        std::cout << "Wrong argument for flag --" << flag_name << ": " << value
                  << "\nHas to be one of: reference, omp, cuda" << std::endl;
        return false;
    }
    return true;
};
DEFINE_validator(executor, validate_executor);


int main(int argc, char *argv[])
{
    gflags::SetUsageMessage("Usage: " + std::string(argv[0]) + " [options]");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::cerr << gko::version_info::get() << std::endl
              << "Running on " << FLAGS_executor << "(" << FLAGS_device_id
              << ")" << std::endl
              << "Running CG with " << FLAGS_max_iters
              << " iterations and residual goal of " << FLAGS_rel_res_goal
              << std::endl;

    auto exec = executor_factory.at(FLAGS_executor)();
}
