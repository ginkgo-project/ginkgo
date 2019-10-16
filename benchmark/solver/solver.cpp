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


#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>


#define DISABLE_FORMATS_COMMAND
#include "benchmark/utils/common.hpp"
#undef DISABLE_FORMATS_COMMAND
#include "benchmark/utils/general.hpp"
#include "benchmark/utils/loggers.hpp"


// some Ginkgo shortcuts
using etype = double;


// Command-line arguments
DEFINE_uint32(max_iters, 1000,
              "Maximal number of iterations the solver will be run for");

DEFINE_double(rel_res_goal, 1e-6, "The relative residual goal of the solver");

DEFINE_string(solvers, "cg",
              "A comma-separated list of solvers to run."
              "Supported values are: bicgstab, cg, cgs, fcg, gmres");

DEFINE_string(preconditioners, "none",
              "A comma-separated list of preconditioners to use."
              "Supported values are: none, jacobi, adaptive-jacobi");

DEFINE_uint32(
    nrhs, 1,
    "The number of right hand sides. Record the residual only when nrhs == 1.");


// input validation
[[noreturn]] void print_config_error_and_exit() {
    std::cerr << "Input has to be a JSON array of matrix configurations:\n"
              << "  [\n"
              << "    { \"filename\": \"my_file.mtx\",  \"optimal\": { "
                 "\"spmv\": \"<matrix format>\" } },\n"
              << "    { \"filename\": \"my_file2.mtx\", \"optimal\": { "
                 "\"spmv\": \"<matrix format>\" } }\n"
              << "  ]" << std::endl;
    std::exit(1);
}


void validate_option_object(const rapidjson::Value &value)
{
    if (!value.IsObject() || !value.HasMember("optimal") ||
        !value["optimal"].HasMember("spmv") ||
        !value["optimal"]["spmv"].IsString() || !value.HasMember("filename") ||
        !value["filename"].IsString()) {
        print_config_error_and_exit();
    }
}


// solver mapping
template <typename SolverType>
std::unique_ptr<gko::LinOpFactory> create_solver(
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const gko::LinOpFactory> precond)
{
    return SolverType::build()
        .with_criteria(gko::stop::ResidualNormReduction<>::build()
                           .with_reduction_factor(FLAGS_rel_res_goal)
                           .on(exec),
                       gko::stop::Iteration::build()
                           .with_max_iters(FLAGS_max_iters)
                           .on(exec))
        .with_preconditioner(give(precond))
        .on(exec);
}


const std::map<std::string, std::function<std::unique_ptr<gko::LinOpFactory>(
                                std::shared_ptr<const gko::Executor>,
                                std::shared_ptr<const gko::LinOpFactory>)>>
    solver_factory{{"bicgstab", create_solver<gko::solver::Bicgstab<>>},
                   {"cg", create_solver<gko::solver::Cg<>>},
                   {"cgs", create_solver<gko::solver::Cgs<>>},
                   {"fcg", create_solver<gko::solver::Fcg<>>},
                   {"gmres", create_solver<gko::solver::Gmres<>>}};


// TODO: Workaround until GPU matrix conversions are implemented
//       The factory will wrap another factory, and make sure that the
//       input operator is copied to the reference executor, and then sent
//       through the generate function
struct ReferenceFactoryWrapper
    : gko::EnablePolymorphicObject<ReferenceFactoryWrapper, gko::LinOpFactory> {
    ReferenceFactoryWrapper(std::shared_ptr<const gko::Executor> exec)
        : gko::EnablePolymorphicObject<ReferenceFactoryWrapper,
                                       gko::LinOpFactory>(exec)
    {}

    ReferenceFactoryWrapper(std::shared_ptr<const gko::LinOpFactory> f)
        : gko::EnablePolymorphicObject<ReferenceFactoryWrapper,
                                       gko::LinOpFactory>(f->get_executor()),
          base_factory{f}
    {}

    std::shared_ptr<const gko::Executor> exec{gko::ReferenceExecutor::create()};
    std::shared_ptr<const gko::LinOpFactory> base_factory;

protected:
    std::unique_ptr<gko::LinOp> generate_impl(
        std::shared_ptr<const gko::LinOp> op) const override
    {
        return base_factory->generate(gko::clone(exec, op));
    }
};


const std::map<std::string, std::function<std::unique_ptr<gko::LinOpFactory>(
                                std::shared_ptr<const gko::Executor>)>>
    precond_factory{
        {"none",
         [](std::shared_ptr<const gko::Executor> exec) {
             return gko::matrix::IdentityFactory<>::create(exec);
         }},
        {"jacobi",
         [](std::shared_ptr<const gko::Executor> exec) {
             std::shared_ptr<const gko::LinOpFactory> f =
                 gko::preconditioner::Jacobi<>::build().on(exec);
             return std::unique_ptr<ReferenceFactoryWrapper>(
                 new ReferenceFactoryWrapper(f));
         }},
        {"adaptive-jacobi", [](std::shared_ptr<const gko::Executor> exec) {
             std::shared_ptr<const gko::LinOpFactory> f =
                 gko::preconditioner::Jacobi<>::build()
                     .with_storage_optimization(
                         gko::precision_reduction::autodetect())
                     .on(exec);
             return std::unique_ptr<ReferenceFactoryWrapper>(
                 new ReferenceFactoryWrapper(f));
         }}};


void write_precond_info(const gko::LinOp *precond,
                        rapidjson::Value &precond_info,
                        rapidjson::MemoryPoolAllocator<> &allocator)
{
    if (const auto jacobi =
            dynamic_cast<const gko::preconditioner::Jacobi<etype> *>(precond)) {
        // extract block sizes
        const auto bdata =
            jacobi->get_parameters().block_pointers.get_const_data();
        add_or_set_member(precond_info, "block_sizes",
                          rapidjson::Value(rapidjson::kArrayType), allocator);
        const auto nblocks = jacobi->get_num_blocks();
        for (auto i = decltype(nblocks){0}; i < nblocks; ++i) {
            precond_info["block_sizes"].PushBack(bdata[i + 1] - bdata[i],
                                                 allocator);
        }

        // extract block precisions
        const auto pdata =
            jacobi->get_parameters()
                .storage_optimization.block_wise.get_const_data();
        if (pdata) {
            add_or_set_member(precond_info, "block_precisions",
                              rapidjson::Value(rapidjson::kArrayType),
                              allocator);
            for (auto i = decltype(nblocks){0}; i < nblocks; ++i) {
                precond_info["block_precisions"].PushBack(
                    static_cast<int>(pdata[i]), allocator);
            }
        }

        // extract condition numbers
        const auto cdata = jacobi->get_conditioning();
        if (cdata) {
            add_or_set_member(precond_info, "block_conditioning",
                              rapidjson::Value(rapidjson::kArrayType),
                              allocator);
            for (auto i = decltype(nblocks){0}; i < nblocks; ++i) {
                precond_info["block_conditioning"].PushBack(cdata[i],
                                                            allocator);
            }
        }
    }
}


void solve_system(const std::string &solver_name,
                  const std::string &precond_name,
                  const char *precond_solver_name,
                  std::shared_ptr<gko::Executor> exec,
                  std::shared_ptr<const gko::LinOp> system_matrix,
                  const vec<etype> *b, const vec<etype> *x,
                  rapidjson::Value &test_case,
                  rapidjson::MemoryPoolAllocator<> &allocator)
{
    try {
        auto &solver_case = test_case["solver"];
        if (!FLAGS_overwrite && solver_case.HasMember(precond_solver_name)) {
            return;
        }

        add_or_set_member(solver_case, precond_solver_name,
                          rapidjson::Value(rapidjson::kObjectType), allocator);
        auto &solver_json = solver_case[precond_solver_name];
        add_or_set_member(solver_json, "recurrent_residuals",
                          rapidjson::Value(rapidjson::kArrayType), allocator);
        add_or_set_member(solver_json, "true_residuals",
                          rapidjson::Value(rapidjson::kArrayType), allocator);
        if (FLAGS_nrhs == 1) {
            auto rhs_norm = compute_norm(lend(b));
            add_or_set_member(solver_json, "rhs_norm", rhs_norm, allocator);
        }
        for (auto stage : {"generate", "apply"}) {
            add_or_set_member(solver_json, stage,
                              rapidjson::Value(rapidjson::kObjectType),
                              allocator);
            add_or_set_member(solver_json[stage], "components",
                              rapidjson::Value(rapidjson::kObjectType),
                              allocator);
        }

        // warm run
        for (unsigned int i = 0; i < FLAGS_warmup; i++) {
            auto x_clone = clone(x);
            auto precond = precond_factory.at(precond_name)(exec);
            auto solver = solver_factory.at(solver_name)(exec, give(precond))
                              ->generate(system_matrix);
            solver->apply(lend(b), lend(x_clone));
            exec->synchronize();
        }

        // detail run
        if (FLAGS_detailed) {
            // slow run, get the time of each functions
            auto x_clone = clone(x);

            auto gen_logger = std::make_shared<OperationLogger>(exec);
            exec->add_logger(gen_logger);

            auto precond = precond_factory.at(precond_name)(exec);
            auto solver = solver_factory.at(solver_name)(exec, give(precond))
                              ->generate(system_matrix);

            exec->remove_logger(gko::lend(gen_logger));
            gen_logger->write_data(solver_json["generate"]["components"],
                                   allocator, 1);

            if (auto prec =
                    dynamic_cast<const gko::Preconditionable *>(lend(solver))) {
                add_or_set_member(solver_json, "preconditioner",
                                  rapidjson::Value(rapidjson::kObjectType),
                                  allocator);
                write_precond_info(lend(clone(get_executor()->get_master(),
                                              prec->get_preconditioner())),
                                   solver_json["preconditioner"], allocator);
            }

            auto apply_logger = std::make_shared<OperationLogger>(exec);
            exec->add_logger(apply_logger);

            solver->apply(lend(b), lend(x_clone));

            exec->remove_logger(gko::lend(apply_logger));
            apply_logger->write_data(solver_json["apply"]["components"],
                                     allocator, 1);

            // slow run, gets the recurrent and true residuals of each iteration
            if (FLAGS_nrhs == 1) {
                x_clone = clone(x);
                auto res_logger = std::make_shared<ResidualLogger<etype>>(
                    exec, lend(system_matrix), b,
                    solver_json["recurrent_residuals"],
                    solver_json["true_residuals"], allocator);
                solver->add_logger(res_logger);
                solver->apply(lend(b), lend(x_clone));
            }
            exec->synchronize();
        }

        // timed run
        std::chrono::nanoseconds apply_time(0);
        std::chrono::nanoseconds generate_time(0);
        for (unsigned int i = 0; i < FLAGS_repetitions; i++) {
            auto x_clone = clone(x);

            exec->synchronize();
            auto g_tic = std::chrono::steady_clock::now();

            auto precond = precond_factory.at(precond_name)(exec);
            auto solver = solver_factory.at(solver_name)(exec, give(precond))
                              ->generate(system_matrix);

            exec->synchronize();
            auto g_tac = std::chrono::steady_clock::now();
            generate_time +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(g_tac -
                                                                     g_tic);

            exec->synchronize();
            auto a_tic = std::chrono::steady_clock::now();

            solver->apply(lend(b), lend(x_clone));

            exec->synchronize();
            auto a_tac = std::chrono::steady_clock::now();
            apply_time += std::chrono::duration_cast<std::chrono::nanoseconds>(
                a_tac - a_tic);

            if (FLAGS_nrhs == 1 && i == FLAGS_repetitions - 1) {
                auto residual = compute_residual_norm(lend(system_matrix),
                                                      lend(b), lend(x_clone));
                add_or_set_member(solver_json, "residual_norm", residual,
                                  allocator);
            }
        }
        add_or_set_member(
            solver_json["generate"], "time",
            static_cast<double>(generate_time.count()) / FLAGS_repetitions,
            allocator);
        add_or_set_member(
            solver_json["apply"], "time",
            static_cast<double>(apply_time.count()) / FLAGS_repetitions,
            allocator);

        // compute and write benchmark data
        add_or_set_member(solver_json, "completed", true, allocator);
    } catch (const std::exception &e) {
        add_or_set_member(test_case["solver"][precond_solver_name], "completed",
                          false, allocator);
        std::cerr << "Error when processing test case " << test_case << "\n"
                  << "what(): " << e.what() << std::endl;
    }
}


int main(int argc, char *argv[])
{
    // Set the default repetitions = 1.
    FLAGS_repetitions = 1;
    std::string header =
        "A benchmark for measuring performance of Ginkgo's solvers.\n";
    std::string format =
        std::string() + "  [\n" +
        "    { \"filename\": \"my_file.mtx\",  \"optimal\": { "
        "\"spmv\": \"<matrix format>\" } },\n" +
        "    { \"filename\": \"my_file2.mtx\", \"optimal\": { "
        "\"spmv\": \"<matrix format>\" } }\n" +
        "  ]\n\n" +
        "  \"optimal_format\" can be one of the recognized spmv "
        "format\n\n";
    initialize_argument_parsing(&argc, &argv, header, format);

    std::string extra_information = "Running " + FLAGS_solvers + " with " +
                                    std::to_string(FLAGS_max_iters) +
                                    " iterations and residual goal of " +
                                    std::to_string(FLAGS_rel_res_goal) +
                                    "\nThe number of right hand sides is " +
                                    std::to_string(FLAGS_nrhs) + "\n";
    print_general_information(extra_information);

    auto exec = get_executor();
    auto solvers = split(FLAGS_solvers, ',');
    auto preconds = split(FLAGS_preconditioners, ',');
    std::vector<std::string> precond_solvers;
    for (const auto &s : solvers) {
        for (const auto &p : preconds) {
            precond_solvers.push_back(s + (p == "none" ? "" : "-" + p));
        }
    }

    rapidjson::IStreamWrapper jcin(std::cin);
    rapidjson::Document test_cases;
    test_cases.ParseStream(jcin);
    if (!test_cases.IsArray()) {
        print_config_error_and_exit();
    }

    auto engine = get_engine();
    auto &allocator = test_cases.GetAllocator();

    for (auto &test_case : test_cases.GetArray()) {
        try {
            // set up benchmark
            validate_option_object(test_case);
            if (!test_case.HasMember("solver")) {
                test_case.AddMember("solver",
                                    rapidjson::Value(rapidjson::kObjectType),
                                    allocator);
            }
            auto &solver_case = test_case["solver"];
            if (!FLAGS_overwrite &&
                all_of(begin(precond_solvers), end(precond_solvers),
                       [&solver_case](const std::string &s) {
                           return solver_case.HasMember(s.c_str());
                       })) {
                continue;
            }
            std::clog << "Running test case: " << test_case << std::endl;
            std::ifstream mtx_fd(test_case["filename"].GetString());
            auto data = gko::read_raw<etype>(mtx_fd);

            auto system_matrix = share(matrix_factory.at(
                test_case["optimal"]["spmv"].GetString())(exec, data));
            auto b = create_matrix<etype>(
                exec, gko::dim<2>{system_matrix->get_size()[0], FLAGS_nrhs},
                engine);
            auto x = create_matrix<etype>(
                exec, gko::dim<2>{system_matrix->get_size()[0], FLAGS_nrhs});

            std::clog << "Matrix is of size (" << system_matrix->get_size()[0]
                      << ", " << system_matrix->get_size()[1] << ")"
                      << std::endl;
            auto precond_solver_name = begin(precond_solvers);
            for (const auto &solver_name : solvers) {
                for (const auto &precond_name : preconds) {
                    std::clog << "\tRunning solver: " << *precond_solver_name
                              << std::endl;
                    solve_system(solver_name, precond_name,
                                 precond_solver_name->c_str(), exec,
                                 system_matrix, lend(b), lend(x), test_case,
                                 allocator);
                    backup_results(test_cases);
                    ++precond_solver_name;
                }
            }
        } catch (const std::exception &e) {
            std::cerr << "Error setting up solver, what(): " << e.what()
                      << std::endl;
        }
    }

    std::cout << test_cases;
}
