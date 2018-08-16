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
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>


#include <gflags/gflags.h>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/prettywriter.h>


// some Ginkgo shortcuts
using vector = gko::matrix::Dense<>;


// helper for writing out rapidjson Values
std::ostream &operator<<(std::ostream &os, const rapidjson::Value &value)
{
    rapidjson::OStreamWrapper jos(os);
    rapidjson::PrettyWriter<rapidjson::OStreamWrapper> writer(jos);
    value.Accept(writer);
    return os;
}


// config validation
void print_config_error_and_exit()
{
    std::cerr
        << "Input has to be a JSON array of matrix configurations:"
        << "[\n    { \"filename\": my_file.mtx,  \"optimal_format\": coo },"
        << "\n    { \"filename\": my_file2.mtx, \"optimal_format\": csr }"
        << "\n]" << std::endl;
    exit(1);
}


void validate_option_object(const rapidjson::Value &value)
{
    if (!value.IsObject() || !value.HasMember("optimal_format") ||
        !value["optimal_format"].IsString() || !value.HasMember("filename") ||
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
        std::cerr << "Wrong argument for flag --" << flag_name << ": " << value
                  << "\nHas to be one of: reference, omp, cuda" << std::endl;
        return false;
    }
    return true;
};
DEFINE_validator(executor, validate_executor);

DEFINE_uint32(rhs_seed, 1234, "Seed used to generate the right hand side");


void initialize_argument_parsing(int *argc, char **argv[])
{
    std::ostringstream doc;
    doc << "A benchmark for measuring performance of Ginkgo's solvers.\n"
        << "Usage: " << (*argv)[0] << "[options]\n"
        << "  The standard input should contain a list of test cases as a JSON"
        << "  array of objects:"
        << "[\n"
        << "    { \"filename\": my_file.mtx,  \"optimal_format\": coo },\n"
        << "    { \"filename\": my_file2.mtx, \"optimal_format\": csr }\n"
        << "]\n\n"
        << "\"optimal_format\" can be one of: \"csr\", \"coo\", \"ell\","
        << "\"hybrid\", \"sellp\"" << std::endl;
    gflags::SetUsageMessage(doc.str());
    std::ostringstream ver;
    ver << gko::version_info::get();
    gflags::SetVersionString(ver.str());
    gflags::ParseCommandLineFlags(argc, argv, true);
}


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


// system solution
template <typename RandomEngine>
std::unique_ptr<vector> create_rhs(std::shared_ptr<const gko::Executor> exec,
                                   RandomEngine &engine, gko::size_type size)
{
    auto rhs = vector::create(exec);
    rhs->read(gko::matrix_data<>(gko::dim<2>{size, 1},
                                 std::uniform_real_distribution<>(-1.0, 1.0),
                                 engine));
    return rhs;
}


std::unique_ptr<vector> create_initial_guess(
    std::shared_ptr<const gko::Executor> exec, gko::size_type size)
{
    auto rhs = vector::create(exec);
    rhs->read(gko::matrix_data<>(gko::dim<2>{size, 1}));
    return rhs;
}


double compute_residual_norm(const gko::LinOp *system_matrix, const vector *b,
                             const vector *x)
{
    auto exec = system_matrix->get_executor();
    auto one = gko::initialize<vector>({1.0}, exec);
    auto neg_one = gko::initialize<vector>({-1.0}, exec);
    auto res_norm = gko::initialize<vector>({0.0}, exec);
    auto res = clone(b);
    system_matrix->apply(lend(one), lend(x), lend(neg_one), lend(res));
    res->compute_dot(lend(res), lend(res_norm));
    return clone(res_norm->get_executor()->get_master(), res_norm)->at(0, 0);
}


double compute_rhs_norm(const vector *b)
{
    auto exec = b->get_executor();
    auto b_norm = gko::initialize<vector>({0.0}, exec);
    b->compute_dot(lend(b), lend(b_norm));
    return clone(exec->get_master(), b_norm)->at(0, 0);
};


template <typename RandomEngine, typename Allocator>
void solve_system(std::shared_ptr<const gko::Executor> exec,
                  rapidjson::Value &test_case, Allocator &allocator,
                  RandomEngine &rhs_engine)
{
    // set up benchmark
    validate_option_object(test_case);

    std::clog << "Running test case: " << test_case << std::endl;

    auto system_matrix = share(matrix_factory.at(
        test_case["optimal_format"].GetString())(exec, test_case));
    auto b = create_rhs(exec, rhs_engine, system_matrix->get_size()[0]);
    auto x = create_initial_guess(exec, system_matrix->get_size()[0]);

    std::clog << "Matrix is of size (" << system_matrix->get_size()[0] << ", "
              << system_matrix->get_size()[1] << ")" << std::endl;

    // TODO: slow run, with logger, to get per-iteration residual info

    // timed run
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

    // compute and write benchmark data
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(tac - tic);
    auto residual =
        compute_residual_norm(lend(system_matrix), lend(b), lend(x));
    auto rhs_norm = compute_rhs_norm(lend(b));

    test_case.AddMember("cg_time", time.count(), allocator);
    test_case.AddMember("cg_completed", true, allocator);
    test_case.AddMember("cg_residual_norm", residual, allocator);
    test_case.AddMember("cg_rhs_norm", rhs_norm, allocator);
};


int main(int argc, char *argv[])
{
    auto format_stream = [](std::ostream &os) {
        os << std::scientific << std::setprecision(16);
    };
    format_stream(std::cout);
    format_stream(std::cerr);
    format_stream(std::clog);

    initialize_argument_parsing(&argc, &argv);

    std::clog << gko::version_info::get() << std::endl
              << "Running on " << FLAGS_executor << "(" << FLAGS_device_id
              << ")" << std::endl
              << "Running CG with " << FLAGS_max_iters
              << " iterations and residual goal of " << FLAGS_rel_res_goal
              << std::endl
              << "The random seed for right hand sides is " << FLAGS_rhs_seed
              << std::endl;

    auto exec = executor_factory.at(FLAGS_executor)();

    rapidjson::IStreamWrapper jcin(std::cin);
    rapidjson::Document test_cases;
    test_cases.ParseStream(jcin);
    if (!test_cases.IsArray()) {
        print_config_error_and_exit();
    }

    std::ranlux24 rhs_engine(FLAGS_rhs_seed);
    auto &allocator = test_cases.GetAllocator();

    for (auto &test_case : test_cases.GetArray()) {
        try {
            solve_system(exec, test_case, allocator, rhs_engine);
        } catch (std::exception &e) {
            test_case.AddMember("cg_completed", false, allocator);
            std::cerr << "Error when processing test case " << test_case << "\n"
                      << "what(): " << e.what() << std::endl;
        }
        std::clog << "Current state:" << std::endl << test_cases << std::endl;
    }
    std::cout << test_cases;
}
