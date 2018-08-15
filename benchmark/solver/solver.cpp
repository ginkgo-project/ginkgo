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
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>


// rapidjson wrapper for cin and cout
rapidjson::IStreamWrapper jcin(std::cin);
rapidjson::OStreamWrapper jcout(std::cout);


// config validation
void print_config_error_and_exit()
{
    std::cerr << "Input has tobe a JSON array of matrix configurations:"
              << "[\n    { \"filename\": my_file.mtx, \"format\": coo },"
              << "\n    {\"filename\": my_file2.mtx, \"format\": csr }"
              << "\n]" << std::endl;
    exit(1);
}


void validate_option_object(const rapidjson::Value &value)
{
    if (!value.IsObject() || !value.HasMember("format") ||
        !value["format"].IsString() || !value.HasMember("filename") ||
        !value["filename"].IsString()) {
        print_config_error_and_exit();
    }
}


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


// matrix format mapping
template <typename MatrixType>
std::unique_ptr<gko::LinOp> read_matrix(
    std::shared_ptr<const gko::Executor> exec, const rapidjson::Value &options)
{
    return gko::read<MatrixType>(std::ifstream(options["filename"].GetString()),
                                 std::move(exec));
}

const std::map<std::string, std::function<std::unique_ptr<gko::LinOp>(
                                std::shared_ptr<const gko::Executor>,
                                const rapidjson::Value &)>>
    matrix_factory{{"csr", read_matrix<gko::matrix::Csr<>>},
                   {"coo", read_matrix<gko::matrix::Coo<>>},
                   {"ell", read_matrix<gko::matrix::Ell<>>},
                   {"hybrid", read_matrix<gko::matrix::Hybrid<>>},
                   {"sellp", read_matrix<gko::matrix::Sellp<>>}};


std::unique_ptr<gko::matrix::Dense<>> create_rhs(
    std::shared_ptr<const gko::Executor> exec, gko::size_type size)
{
    auto rhs = gko::matrix::Dense<>::create(exec);
    rhs->read(gko::matrix_data<>(gko::dim<2>{size, 1}, 1.0));
    return rhs;
}

std::unique_ptr<gko::matrix::Dense<>> create_initial_guess(
    std::shared_ptr<const gko::Executor> exec, gko::size_type size)
{
    auto rhs = gko::matrix::Dense<>::create(exec);
    rhs->read(gko::matrix_data<>(gko::dim<2>{size, 1}));
    return rhs;
}

void solve_system(std::shared_ptr<const gko::Executor> exec,
                  const rapidjson::Value &test_case)
{
    validate_option_object(test_case);

    rapidjson::Writer<rapidjson::OStreamWrapper> writer(jcout);
    std::cout << "Running test case: ";
    test_case.Accept(writer);
    std::cout << std::endl;

    auto system_matrix = share(
        matrix_factory.at(test_case["format"].GetString())(exec, test_case));
    auto b = create_rhs(exec, system_matrix->get_size()[0]);
    auto x = create_initial_guess(exec, system_matrix->get_size()[0]);

    std::cout << "Matrix is of size (" << system_matrix->get_size()[0] << ", "
              << system_matrix->get_size()[1] << ")" << std::endl;

    exec->synchronize();
    auto tic = std::chrono::system_clock::now();

    auto solver =
        gko::solver::Cg<>::Factory::create()
            .with_criterion(
                gko::stop::Combined::Factory::create()
                    .with_criteria(
                        gko::stop::ResidualNormReduction<>::Factory::create()
                            .with_reduction_factor(FLAGS_rel_res_goal)
                            .on_executor(exec),
                        gko::stop::Iteration::Factory::create()
                            .with_max_iters(FLAGS_max_iters)
                            .on_executor(exec))
                    .on_executor(exec))
            .on_executor(exec);

    solver->generate(system_matrix)->apply(lend(b), lend(x));

    exec->synchronize();
    auto tac = std::chrono::system_clock::now();

    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(tac - tic);

    std::cout << "Time: " << time.count() << std::endl;
};


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

    rapidjson::Document config;
    config.ParseStream(jcin);

    if (!config.IsArray()) {
        print_config_error_and_exit();
    }
    for (auto &test_case : config.GetArray()) {
        solve_system(exec, test_case);
    }
}
