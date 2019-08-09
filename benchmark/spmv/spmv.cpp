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
#include <typeinfo>


#include "benchmark/utils/general.hpp"
#include "benchmark/utils/loggers.hpp"
#include "benchmark/utils/spmv_common.hpp"


#ifdef HAS_CUDA
#include "cuda_linops.hpp"
#endif  // HAS_CUDA


using etype = double;


// Command-line arguments
DEFINE_string(
    formats, "coo",
    "A comma-separated list of formats to run."
    "Supported values are: coo, csr, ell, sellp, hybrid, hybrid0, "
    "hybrid25, hybrid33, hybrid40, hybrid60, hybrid80, hybridlimit0, "
    "hybridlimit25, hybridlimit33, hybridminstorage, cusp_csr, cusp_csrex, "
    "cusp_csrmp, cusp_csrmm, cusp_coo, cusp_ell, cusp_hybrid.\n"
    "coo: Coordinate storage. The CUDA kernel uses the load-balancing approach "
    "suggested in Flegar et al.: Overcoming Load Imbalance for Irregular "
    "Sparse Matrices.\n"
    "csr: Compressed Sparse Row storage. Ginkgo implementation with automatic "
    "strategy.\n"
    "csrc: Ginkgo's CSR implementation with automatic stategy.\n"
    "csri: Ginkgo's CSR implementation with inbalance strategy.\n"
    "csrm: Ginkgo's CSR implementation with merge_path strategy.\n"
    "ell: Ellpack format according to Bell and Garland: Efficient Sparse "
    "Matrix-Vector Multiplication on CUDA.\n"
    "sellp: Sliced Ellpack uses a default block size of 32.\n"
    "hybrid: Hybrid uses ell and coo to represent the matrix.\n"
    "hybrid0, hybrid25, hybrid33, hybrid40, hybrid60, hybrid80: Hybrid uses "
    "the row distribution to decide the partition.\n"
    "hybridlimit0, hybridlimit25, hybrid33: Add the upper bound on the ell "
    "part of hybrid0, hybrid25, hybrid33.\n"
    "hybridminstorage: Hybrid uses the minimal storage to store the matrix.\n"
    "cusp_hybrid: benchmark CuSPARSE spmv with cusparseXhybmv and an automatic "
    "partition.\n"
    "cusp_coo: use cusparseXhybmv with a CUSPARSE_HYB_PARTITION_USER "
    "partition.\n"
    "cusp_ell: use cusparseXhybmv with CUSPARSE_HYB_PARTITION_MAX partition.\n"
    "cusp_csr: benchmark CuSPARSE with the cusparseXcsrmv function.\n"
    "cusp_csrex: benchmark CuSPARSE with the cusparseXcsrmvEx function.\n"
    "cusp_csrmp: benchmark CuSPARSE with the cusparseXcsrmv_mp function.\n"
    "cusp_csrmm: benchmark CuSPARSE with the cusparseXcsrmv_mm function.\n");

DEFINE_uint32(nrhs, 1, "The number of right hand sides");


const std::map<std::string, std::function<std::unique_ptr<gko::LinOp>(
                                std::shared_ptr<const gko::Executor>,
                                const gko::matrix_data<> &)>>
    matrix_factory{
        {"csr", READ_MATRIX(csr, std::make_shared<csr::automatical>())},
        {"csri", READ_MATRIX(csr, std::make_shared<csr::load_balance>())},
        {"csrm", READ_MATRIX(csr, std::make_shared<csr::merge_path>())},
        {"csrc", READ_MATRIX(csr, std::make_shared<csr::classical>())},
        {"coo", read_matrix_from_data<gko::matrix::Coo<>>},
        {"ell", read_matrix_from_data<gko::matrix::Ell<>>},
#ifdef HAS_CUDA
        {"cusp_csr", read_matrix_from_data<cusp_csr>},
        {"cusp_csrmp", read_matrix_from_data<cusp_csrmp>},
        {"cusp_csrex", read_matrix_from_data<cusp_csrex>},
        {"cusp_csrmm", read_matrix_from_data<cusp_csrmm>},
        {"cusp_hybrid", read_matrix_from_data<cusp_hybrid>},
        {"cusp_coo", read_matrix_from_data<cusp_coo>},
        {"cusp_ell", read_matrix_from_data<cusp_ell>},
#endif  // HAS_CUDA
        {"hybrid", read_matrix_from_data<hybrid>},
        {"hybrid0",
         READ_MATRIX(hybrid, std::make_shared<hybrid::imbalance_limit>(0))},
        {"hybrid25",
         READ_MATRIX(hybrid, std::make_shared<hybrid::imbalance_limit>(0.25))},
        {"hybrid33",
         READ_MATRIX(hybrid,
                     std::make_shared<hybrid::imbalance_limit>(1.0 / 3.0))},
        {"hybrid40",
         READ_MATRIX(hybrid, std::make_shared<hybrid::imbalance_limit>(0.4))},
        {"hybrid60",
         READ_MATRIX(hybrid, std::make_shared<hybrid::imbalance_limit>(0.6))},
        {"hybrid80",
         READ_MATRIX(hybrid, std::make_shared<hybrid::imbalance_limit>(0.8))},
        {"hybridlimit0",
         READ_MATRIX(hybrid,
                     std::make_shared<hybrid::imbalance_bounded_limit>(0))},
        {"hybridlimit25",
         READ_MATRIX(hybrid,
                     std::make_shared<hybrid::imbalance_bounded_limit>(0.25))},
        {"hybridlimit33",
         READ_MATRIX(hybrid, std::make_shared<hybrid::imbalance_bounded_limit>(
                                 1.0 / 3.0))},
        {"hybridminstorage",
         READ_MATRIX(hybrid,
                     std::make_shared<hybrid::minimal_storage_limit>())},
        {"sellp", read_matrix_from_data<gko::matrix::Sellp<>>}};


// This function supposes that management of `FLAGS_overwrite` is done before
// calling it
void apply_spmv(const char *format_name, std::shared_ptr<gko::Executor> exec,
                const gko::matrix_data<etype> &data, const vec<etype> *b,
                const vec<etype> *x, rapidjson::Value &test_case,
                rapidjson::MemoryPoolAllocator<> &allocator)
{
    try {
        auto &spmv_case = test_case["spmv"];
        add_or_set_member(spmv_case, format_name,
                          rapidjson::Value(rapidjson::kObjectType), allocator);

        auto storage_logger = std::make_shared<StorageLogger>(exec);
        exec->add_logger(storage_logger);
        auto system_matrix = share(matrix_factory.at(format_name)(exec, data));
        exec->remove_logger(gko::lend(storage_logger));
        storage_logger->write_data(spmv_case[format_name], allocator);
        // warm run
        for (unsigned int i = 0; i < FLAGS_warmup; i++) {
            auto x_clone = clone(x);
            exec->synchronize();
            system_matrix->apply(lend(b), lend(x_clone));
            exec->synchronize();
        }
        std::chrono::nanoseconds time(0);
        // timed run
        for (unsigned int i = 0; i < FLAGS_repetitions; i++) {
            auto x_clone = clone(x);
            exec->synchronize();
            auto tic = std::chrono::steady_clock::now();
            system_matrix->apply(lend(b), lend(x_clone));

            exec->synchronize();
            auto toc = std::chrono::steady_clock::now();
            time +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);
        }
        add_or_set_member(spmv_case[format_name], "time",
                          static_cast<double>(time.count()) / FLAGS_repetitions,
                          allocator);

        // compute and write benchmark data
        add_or_set_member(spmv_case[format_name], "completed", true, allocator);
    } catch (const std::exception &e) {
        add_or_set_member(test_case["spmv"][format_name], "completed", false,
                          allocator);
        std::cerr << "Error when processing test case " << test_case << "\n"
                  << "what(): " << e.what() << std::endl;
    }
}


int main(int argc, char *argv[])
{
    std::string header =
        "A benchmark for measuring performance of Ginkgo's spmv.\n";
    std::string format = std::string() + "  [\n" +
                         "    { \"filename\": \"my_file.mtx\"},\n" +
                         "    { \"filename\": \"my_file2.mtx\"}\n" + "  ]\n\n";
    initialize_argument_parsing(&argc, &argv, header, format);

    std::string extra_information = "The formats are " + FLAGS_formats +
                                    "\nThe number of right hand sides is " +
                                    std::to_string(FLAGS_nrhs) + "\n";
    print_general_information(extra_information);

    auto exec = executor_factory.at(FLAGS_executor)();
    auto engine = get_engine();
    auto formats = split(FLAGS_formats, ',');

    rapidjson::IStreamWrapper jcin(std::cin);
    rapidjson::Document test_cases;
    test_cases.ParseStream(jcin);
    if (!test_cases.IsArray()) {
        print_config_error_and_exit();
    }

    auto &allocator = test_cases.GetAllocator();

    for (auto &test_case : test_cases.GetArray()) {
        try {
            // set up benchmark
            validate_option_object(test_case);
            if (!test_case.HasMember("spmv")) {
                test_case.AddMember("spmv",
                                    rapidjson::Value(rapidjson::kObjectType),
                                    allocator);
            }
            auto &spmv_case = test_case["spmv"];
            if (!FLAGS_overwrite &&
                all_of(begin(formats), end(formats),
                       [&spmv_case](const std::string &s) {
                           return spmv_case.HasMember(s.c_str());
                       })) {
                continue;
            }
            std::clog << "Running test case: " << test_case << std::endl;
            std::ifstream mtx_fd(test_case["filename"].GetString());
            auto data = gko::read_raw<etype>(mtx_fd);

            auto nrhs = FLAGS_nrhs;
            auto b = create_matrix<etype>(exec, gko::dim<2>{data.size[1], nrhs},
                                          engine);
            auto x = create_matrix<etype>(exec, gko::dim<2>{data.size[0], nrhs},
                                          engine);
            std::clog << "Matrix is of size (" << data.size[0] << ", "
                      << data.size[1] << ")" << std::endl;
            std::string best_format("none");
            auto best_performance = 0.0;
            if (!test_case.HasMember("optimal")) {
                test_case.AddMember("optimal",
                                    rapidjson::Value(rapidjson::kObjectType),
                                    allocator);
            }
            for (const auto &format_name : formats) {
                apply_spmv(format_name.c_str(), exec, data, lend(b), lend(x),
                           test_case, allocator);
                std::clog << "Current state:" << std::endl
                          << test_cases << std::endl;
                if (spmv_case[format_name.c_str()]["completed"].GetBool()) {
                    auto performance =
                        spmv_case[format_name.c_str()]["time"].GetDouble();
                    if (best_format == "none" ||
                        performance < best_performance) {
                        best_format = format_name;
                        best_performance = performance;
                        add_or_set_member(
                            test_case["optimal"], "spmv",
                            rapidjson::Value(best_format.c_str(), allocator)
                                .Move(),
                            allocator);
                    }
                }
                backup_results(test_cases);
            }
        } catch (const std::exception &e) {
            std::cerr << "Error setting up matrix data, what(): " << e.what()
                      << std::endl;
        }
    }

    std::cout << test_cases;
}
