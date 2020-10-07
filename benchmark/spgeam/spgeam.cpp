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
#include <numeric>
#include <random>
#include <typeinfo>
#include <unordered_set>


#include "benchmark/utils/general.hpp"
#include "benchmark/utils/loggers.hpp"
#include "benchmark/utils/spmv_common.hpp"
#include "core/test/utils/matrix_generator.hpp"
#include "ginkgo/core/matrix/dense.hpp"

using etype = double;
#ifdef GKO_SPGEAM_LONG
using itype = gko::int64;
const auto benchmark_name = "spgeam64";
#else
using itype = gko::int32;
const auto benchmark_name = "spgeam";
#endif
using Mtx = gko::matrix::Csr<etype, itype>;
using Dense = gko::matrix::Dense<etype>;
using mat_data = gko::matrix_data<etype, itype>;

const std::map<std::string,
               const std::function<std::shared_ptr<Mtx::strategy_type>()>>
    strategy_map{
        {"classical", [] { return std::make_shared<Mtx::classical>(); }},
        {"sparselib", [] { return std::make_shared<Mtx::sparselib>(); }}};

DEFINE_string(
    strategies, "classical,sparselib",
    "Comma-separated list of SpGEAM strategies: classical, sparselib");

DEFINE_int32(
    swap_distance, 100,
    "Maximum distance for row swaps to avoid rows with disjoint column ranges");

std::shared_ptr<Mtx> build_mtx2(std::shared_ptr<gko::Executor> exec,
                                gko::matrix_data<etype, itype> data)
{
    // randomly permute n/2 rows with limited distances
    std::vector<itype> permutation(data.size[0]);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::default_random_engine rng(FLAGS_seed);
    std::uniform_int_distribution<itype> start_dist(0, data.size[0] - 1);
    std::uniform_int_distribution<itype> delta_dist(-FLAGS_swap_distance,
                                                    FLAGS_swap_distance);
    for (itype i = 0; i < data.size[0] / 2; ++i) {
        auto a = start_dist(rng);
        auto b = a + delta_dist(rng);
        if (b >= 0 && b < data.size[0]) {
            std::swap(permutation[a], permutation[b]);
        }
    }
    for (auto &nonzero : data.nonzeros) {
        nonzero.row = permutation[nonzero.row];
    }
    data.ensure_row_major_order();
    auto result = Mtx::create(exec, data.size, data.nonzeros.size());
    result->read(data);
    return result;
}

void apply_spgeam(const char *strategy_name,
                  std::shared_ptr<gko::Executor> exec,
                  const gko::matrix_data<etype, itype> &data, const Mtx *mtx2,
                  rapidjson::Value &test_case,
                  rapidjson::MemoryPoolAllocator<> &allocator)
{
    try {
        add_or_set_member(test_case, strategy_name,
                          rapidjson::Value(rapidjson::kObjectType), allocator);

        auto mtx = Mtx::create(exec, data.size, data.nonzeros.size(),
                               strategy_map.at(strategy_name)());
        mtx->read(data);
        auto one = gko::initialize<Dense>({1.0}, exec);
        auto id =
            gko::matrix::Identity<etype>::create(exec, mtx->get_size()[0]);
        auto res = Mtx::create(exec, gko::dim<2>(mtx->get_size()));

        // warm run
        for (unsigned int i = 0; i < FLAGS_warmup; i++) {
            res->copy_from(mtx2);
            exec->synchronize();
            mtx->apply(lend(one), lend(id), lend(one), lend(res));
            exec->synchronize();
        }

        std::chrono::nanoseconds time(0);
        // timed run
        for (unsigned int i = 0; i < FLAGS_repetitions; i++) {
            res->copy_from(mtx2);
            exec->synchronize();
            auto tic = std::chrono::steady_clock::now();
            mtx->apply(lend(one), lend(id), lend(one), lend(res));

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
                res->copy_from(mtx2);
                mtx->apply(lend(one), lend(id), lend(one), lend(res));
            }
            exec->remove_logger(gko::lend(op_logger));

            op_logger->write_data(test_case[strategy_name]["components"],
                                  allocator, FLAGS_repetitions);
        }

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
            if (!test_case.HasMember(benchmark_name)) {
                test_case.AddMember(rapidjson::Value(benchmark_name, allocator),
                                    rapidjson::Value(rapidjson::kObjectType),
                                    allocator);
            }
            auto &spgeam_case = test_case[benchmark_name];
            if (!FLAGS_overwrite &&
                all_of(begin(strategies), end(strategies),
                       [&](const std::string &s) {
                           return spgeam_case.HasMember(s.c_str());
                       })) {
                continue;
            }
            std::clog << "Running test case: " << test_case << std::endl;
            std::ifstream mtx_fd(test_case["filename"].GetString());
            auto data = gko::read_raw<etype, itype>(mtx_fd);
            data.ensure_row_major_order();
            auto mtx2 = build_mtx2(exec, data);

            std::clog << "Matrix is of size (" << data.size[0] << ", "
                      << data.size[1] << "), " << data.nonzeros.size()
                      << std::endl;

            for (const auto &strategy_name : strategies) {
                apply_spgeam(strategy_name.c_str(), exec, data, lend(mtx2),
                             spgeam_case, allocator);
                std::clog << "Current state:" << std::endl
                          << test_cases << std::endl;
                backup_results(test_cases);
            }
            // write the output if we have no strategies
            backup_results(test_cases);
        } catch (const std::exception &e) {
            std::cerr << "Error setting up matrix data, what(): " << e.what()
                      << std::endl;
        }
    }

    std::cout << test_cases << std::endl;
}
