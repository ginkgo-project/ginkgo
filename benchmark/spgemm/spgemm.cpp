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


#include "benchmark/utils/general.hpp"
#include "benchmark/utils/loggers.hpp"
#include "benchmark/utils/spmv_common.hpp"
#include "core/test/utils/matrix_generator.hpp"


using etype = double;
using itype = gko::int32;
using Mtx = gko::matrix::Csr<etype, itype>;

const std::map<std::string,
               const std::function<std::shared_ptr<Mtx::strategy_type>(
                   std::shared_ptr<gko::Executor>)>>
    strategy_map{{"onepass",
                  [](std::shared_ptr<gko::Executor> exec) {
                      // prevent double-free on executors/sparselib handles
                      if (dynamic_cast<gko::HipExecutor *>(exec.get())) {
                          return std::make_shared<Mtx::load_balance>(
                              gko::as<gko::HipExecutor>(exec));
                      }
                      if (dynamic_cast<gko::CudaExecutor *>(exec.get())) {
                          return std::make_shared<Mtx::load_balance>(
                              gko::as<gko::CudaExecutor>(exec));
                      }
                      return std::make_shared<Mtx::load_balance>();
                  }},
                 {"twopass",
                  [](std::shared_ptr<gko::Executor>) {
                      return std::make_shared<Mtx::classical>();
                  }},
                 {"sparselib", [](std::shared_ptr<gko::Executor>) {
                      return std::make_shared<Mtx::sparselib>();
                  }}};


DEFINE_int32(rowlength, 10,
             "The length of rows in randomly generated matrices B. Only "
             "relevant for mode = <sparse|dense>");


const std::map<std::string,
               const std::function<std::shared_ptr<Mtx>(std::shared_ptr<Mtx>)>>
    mode_map{
        {"normal",
         [](std::shared_ptr<Mtx> matrix) {
             return matrix->get_size()[0] == matrix->get_size()[1]
                        ? matrix
                        : gko::as<Mtx>(matrix->transpose());
         }},
        {"transposed",
         [](std::shared_ptr<Mtx> matrix) {
             return gko::as<Mtx>(matrix->transpose());
         }},
        {"sparse",
         [](std::shared_ptr<Mtx> matrix) {
             auto size = gko::transpose(matrix->get_size());
             // don't expect too much quality from this seed =)
             std::default_random_engine rng(
                 FLAGS_seed ^ (matrix->get_size()[0] << 24) ^
                 (matrix->get_size()[1] << 15) -
                     matrix->get_num_stored_elements());
             std::uniform_real_distribution<etype> val_dist(-1.0, 1.0);
             gko::matrix_data<etype, itype> data{size, {}};
             data.nonzeros.reserve(size[0] * FLAGS_rowlength);
             // randomly permute column indices
             std::vector<itype> cols(size[1]);
             std::iota(cols.begin(), cols.end(), 0);
             for (itype row = 0; row < size[0]; ++row) {
                 std::shuffle(cols.begin(), cols.end(), rng);
                 for (int i = 0; i < FLAGS_rowlength; ++i) {
                     data.nonzeros.emplace_back(row, cols[i], val_dist(rng));
                 }
             }
             auto mtx = Mtx::create(matrix->get_executor(), size);
             mtx->read(data);
             return gko::share(std::move(mtx));
         }},
        {"dense", [](std::shared_ptr<Mtx> matrix) {
             auto size = gko::dim<2>(matrix->get_size()[1], FLAGS_rowlength);
             // don't expect too much quality from this seed =)
             std::default_random_engine rng(
                 FLAGS_seed ^ (matrix->get_size()[0] << 24) ^
                 (matrix->get_size()[1] << 15) -
                     matrix->get_num_stored_elements());
             std::uniform_real_distribution<etype> dist(-1.0, 1.0);
             gko::matrix_data<etype, itype> data{size, dist, rng};
             data.ensure_row_major_order();
             auto mtx = Mtx::create(matrix->get_executor(), size);
             mtx->read(data);
             return gko::share(std::move(mtx));
         }}};


DEFINE_string(
    mode, "normal",
    "Which matrix B should be used to compute A * B: normal, "
    "transposed, sparse, dense\n"
    "normal: B = A for A square, A^T otherwise\ntransposed: B = "
    "A^T\nsparse: B is a sparse matrix with dimensions of A^T with uniformly "
    "random values, at most -rowlength non-zeros per row\ndense: B is a "
    "'dense' sparse matrix with -rowlength columns and non-zeros per row");


DEFINE_string(
    strategies, "onepass,twopass,sparselib",
    "Comma-separated list of SpGEMM strategies: onepass, twopass, sparselib");


void apply_spgemm(const char *strategy_name,
                  std::shared_ptr<gko::Executor> exec,
                  const gko::matrix_data<etype> &data,
                  rapidjson::Value &test_case,
                  rapidjson::MemoryPoolAllocator<> &allocator)
{
    try {
        add_or_set_member(test_case, strategy_name,
                          rapidjson::Value(rapidjson::kObjectType), allocator);
        add_or_set_member(test_case[strategy_name], "mode",
                          rapidjson::Value(FLAGS_mode.c_str(), allocator),
                          allocator);

        auto mtx = gko::share(Mtx::create(exec));
        mtx->read(data);
        mtx->set_strategy(strategy_map.at(strategy_name)(exec));
        auto mtx2 = mode_map.at(FLAGS_mode)(mtx);
        auto res = Mtx::create(
            exec, gko::dim<2>(mtx->get_size()[0], mtx2->get_size()[1]));

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
