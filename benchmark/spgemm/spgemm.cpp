/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <typeinfo>


#include "benchmark/utils/general.hpp"
#include "benchmark/utils/loggers.hpp"
#include "benchmark/utils/spmv_common.hpp"


using etype = double;
using itype = gko::int32;


const std::map<std::string,
               const std::function<std::shared_ptr<
                   gko::matrix::Csr<etype, itype>::strategy_type>()>>
    strategy_map{{"classical",
                  []() {
                      return std::make_shared<
                          gko::matrix::Csr<etype, itype>::classical>();
                  }},
                 {"sparselib", []() {
                      return std::make_shared<
                          gko::matrix::Csr<etype, itype>::sparselib>();
                  }}};


DEFINE_bool(transpose_square, false,
            "Compute A*A^T instead of A*A if A is square");

DEFINE_string(
    strategies, "classical,sparselib",
    "Comma-separated list of SpGEMM strategies: classical or sparselib");


void apply_spgemm(const char *strategy_name,
                  std::shared_ptr<gko::Executor> exec,
                  const gko::matrix_data<etype> &data,
                  rapidjson::Value &test_case,
                  rapidjson::MemoryPoolAllocator<> &allocator)
{
    try {
        add_or_set_member(test_case, strategy_name,
                          rapidjson::Value(rapidjson::kObjectType), allocator);

        using Mtx = gko::matrix::Csr<etype, itype>;

        auto mtx = Mtx::create(exec);
        mtx->read(data);
        mtx->set_strategy(strategy_map.at(strategy_name)());
        auto is_square = mtx->get_size()[0] != mtx->get_size()[1];
        auto do_transpose = !is_square || FLAGS_transpose_square;
        auto mtx2 =
            do_transpose ? gko::as<Mtx>(mtx->transpose()) : mtx->clone();
        auto res = Mtx::create(
            exec, gko::dim<2>(mtx->get_size()[0], mtx->get_size()[1]));

        // warm run
        for (unsigned int i = 0; i < FLAGS_warmup; i++) {
            exec->synchronize();
            mtx->apply(lend(mtx2), lend(res));
            exec->synchronize();
        }
        std::chrono::nanoseconds time(0);
        // timed run
        for (unsigned int i = 0; i < FLAGS_repetitions; i++) {
            res = Mtx::create(exec, res->get_size());
            exec->synchronize();
            auto tic = std::chrono::steady_clock::now();
            mtx->apply(lend(mtx2), lend(res));

            exec->synchronize();
            auto toc = std::chrono::steady_clock::now();
            time +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);
        }
        add_or_set_member(test_case[strategy_name], "time",
                          static_cast<double>(time.count()) / FLAGS_repetitions,
                          allocator);

        if (FLAGS_detailed) {
            // slow run, times each component separately
            add_or_set_member(test_case[strategy_name], "components",
                              rapidjson::Value(rapidjson::kObjectType),
                              allocator);

            auto op_logger = std::make_shared<OperationLogger>(exec, true);
            exec->add_logger(op_logger);
            for (auto i = 0u; i < FLAGS_repetitions; ++i) {
                res = Mtx::create(exec, res->get_size());
                mtx->apply(lend(mtx2), lend(res));
            }
            exec->remove_logger(gko::lend(op_logger));

            op_logger->write_data(test_case[strategy_name]["components"],
                                  allocator, FLAGS_repetitions);
        }

        // compute and write benchmark data
        add_or_set_member(test_case[strategy_name], "completed", true,
                          allocator);
    } catch (const std::exception &e) {
        add_or_set_member(test_case[strategy_name], "completed", false,
                          allocator);
        std::cerr << "Error when processing test case " << test_case << "\n"
                  << "what(): " << e.what() << std::endl;
    }
}


int main(int argc, char *argv[])
{
    std::string header =
        "A benchmark for measuring performance of Ginkgo's spgemm.\n";
    std::string format = std::string() + "  [\n" +
                         "    { \"filename\": \"my_file.mtx\"},\n" +
                         "    { \"filename\": \"my_file2.mtx\"}\n" + "  ]\n\n";
    initialize_argument_parsing(&argc, &argv, header, format);

    auto exec = executor_factory.at(FLAGS_executor)();
    auto engine = get_engine();

    rapidjson::IStreamWrapper jcin(std::cin);
    rapidjson::Document test_cases;
    test_cases.ParseStream(jcin);
    if (!test_cases.IsArray()) {
        print_config_error_and_exit();
    }

    print_general_information("");

    auto &allocator = test_cases.GetAllocator();

    auto strategies = split(FLAGS_strategies, ',');

    for (auto &test_case : test_cases.GetArray()) {
        try {
            // set up benchmark
            validate_option_object(test_case);
            if (!test_case.HasMember("spgemm")) {
                test_case.AddMember("spgemm",
                                    rapidjson::Value(rapidjson::kObjectType),
                                    allocator);
            }
            auto &spgemm_case = test_case["spgemm"];
            if (!FLAGS_overwrite &&
                all_of(begin(strategies), end(strategies),
                       [&](const std::string &s) {
                           return spgemm_case.HasMember(s.c_str());
                       })) {
                continue;
            }
            std::clog << "Running test case: " << test_case << std::endl;
            std::ifstream mtx_fd(test_case["filename"].GetString());
            auto data = gko::read_raw<etype>(mtx_fd);

            std::clog << "Matrix is of size (" << data.size[0] << ", "
                      << data.size[1] << ")" << std::endl;

            for (const auto &strategy_name : strategies) {
                apply_spgemm(strategy_name.c_str(), exec, data, spgemm_case,
                             allocator);
                std::clog << "Current state:" << std::endl
                          << test_cases << std::endl;
                backup_results(test_cases);
            }
        } catch (const std::exception &e) {
            std::cerr << "Error setting up matrix data, what(): " << e.what()
                      << std::endl;
        }
    }

    std::cout << test_cases << std::endl;
}
