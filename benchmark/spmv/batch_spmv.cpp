/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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


#include "benchmark/utils/formats.hpp"
#include "benchmark/utils/general.hpp"
#include "benchmark/utils/loggers.hpp"
#include "benchmark/utils/spmv_common.hpp"
#include "benchmark/utils/timer.hpp"
#include "benchmark/utils/types.hpp"


#ifdef GINKGO_BENCHMARK_ENABLE_TUNING
#include "benchmark/utils/tuning_variables.hpp"
#endif  // GINKGO_BENCHMARK_ENABLE_TUNING


// Command-line arguments
DEFINE_uint32(nrhs, 1, "The number of right hand sides");
DEFINE_uint32(num_duplications, 1, "The number of duplications");
DEFINE_uint32(num_batches, 1, "The number of batch entries");
DEFINE_string(batch_scaling, "none", "Whether to use scaled matrices");
DEFINE_bool(using_suite_sparse, true,
            "Whether the suitesparse matrices are being used");
DEFINE_string(
    rhs_generation, "none",
    "Method used to generate the right hand side. Supported values are:"
    "`1`, `random`, `sinus`, `file` . `1` sets all values of the right hand "
    "side to 1, "
    "`random` assigns the values to a uniformly distributed random number "
    "in [-1, 1), `sinus` assigns b = A * (s / |s|) with A := system matrix,"
    " s := vector with s(idx) = sin(idx) for non-complex types, and "
    "s(idx) = sin(2*idx) + i * sin(2*idx+1) and `file` read the rhs from a "
    "file.");


template <typename ValueType>
using batch_vec = gko::matrix::BatchDense<ValueType>;

using size_type = gko::size_type;

// This function supposes that management of `FLAGS_overwrite` is done before
// calling it
//
// Overload for the Suitesparse matrices
void apply_spmv(const char* format_name, std::shared_ptr<gko::Executor> exec,
                const gko::matrix_data<etype>& data, const batch_vec<etype>* b,
                const batch_vec<etype>* x, const batch_vec<etype>* answer,
                rapidjson::Value& test_case,
                rapidjson::MemoryPoolAllocator<>& allocator)
{
    try {
        auto& spmv_case = test_case["spmv"];
        add_or_set_member(spmv_case, format_name,
                          rapidjson::Value(rapidjson::kObjectType), allocator);

        auto nbatch = FLAGS_num_duplications;
        auto storage_logger = std::make_shared<StorageLogger>();
        exec->add_logger(storage_logger);
        std::shared_ptr<gko::BatchLinOp> system_matrix =
            (formats::batch_matrix_factory.at(format_name)(exec, nbatch, data));

        std::clog << "Batch Matrix has: "
                  << system_matrix->get_num_batch_entries()
                  << " batches, each of size ("
                  << system_matrix->get_size().at(0)[0] << ", "
                  << system_matrix->get_size().at(0)[1] << ")" << std::endl;

        exec->remove_logger(gko::lend(storage_logger));
        storage_logger->write_data(spmv_case[format_name], allocator);
        // check the residual
        if (FLAGS_detailed) {
            auto x_clone = clone(x);
            exec->synchronize();
            system_matrix->apply(lend(b), lend(x_clone));
            exec->synchronize();
            auto max_relative_norm2 =
                compute_batch_max_relative_norm2(lend(x_clone), lend(answer));
            // FIXME
            // add_or_set_member(spmv_case[format_name], "max_relative_norm2",
            //                   max_relative_norm2, allocator);
        }
        // warm run
        for (unsigned int i = 0; i < FLAGS_warmup; i++) {
            auto x_clone = clone(x);
            exec->synchronize();
            system_matrix->apply(lend(b), lend(x_clone));
            exec->synchronize();
        }

        IterationControl ic{get_timer(exec, FLAGS_gpu_timer)};

        // tuning run
#ifdef GINKGO_BENCHMARK_ENABLE_TUNING
        auto& format_case = spmv_case[format_name];
        if (!format_case.HasMember("tuning")) {
            format_case.AddMember(
                "tuning", rapidjson::Value(rapidjson::kObjectType), allocator);
        }
        auto& tuning_case = format_case["tuning"];
        add_or_set_member(tuning_case, "time",
                          rapidjson::Value(rapidjson::kArrayType), allocator);
        add_or_set_member(tuning_case, "values",
                          rapidjson::Value(rapidjson::kArrayType), allocator);

        // Enable tuning for this portion of code
        gko::_tuning_flag = true;
        // Select some values we want to tune.
        std::vector<gko::size_type> tuning_values{
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
        for (auto val : tuning_values) {
            // Actually set the value that will be tuned. See
            // cuda/components/format_conversion.cuh for an example of how this
            // variable is used.
            gko::_tuned_value = val;
            auto tuning_timer = get_timer(exec, FLAGS_gpu_timer);
            for (auto status : ic.run(false)) {
                auto x_clone = clone(x);
                exec->synchronize();
                tuning_timer->tic();
                system_matrix->apply(lend(b), lend(x_clone));
                tuning_timer->toc();
            }
            tuning_case["time"].PushBack(tuning_timer->compute_average_time(),
                                         allocator);
            tuning_case["values"].PushBack(val, allocator);
        }
        // We put back the flag to false to use the default (non-tuned) values
        // for the following
        gko::_tuning_flag = false;
#endif  // GINKGO_BENCHMARK_ENABLE_TUNING

        // timed run
        auto timer = get_timer(exec, FLAGS_gpu_timer);
        for (auto status : ic.run(false)) {
            auto x_clone = clone(x);
            exec->synchronize();
            timer->tic();
            system_matrix->apply(lend(b), lend(x_clone));
            timer->toc();
        }
        add_or_set_member(spmv_case[format_name], "num_batch_entries",
                          system_matrix->get_num_batch_entries(), allocator);
        add_or_set_member(spmv_case[format_name], "time",
                          timer->compute_average_time(), allocator);

        // compute and write benchmark data
        add_or_set_member(spmv_case[format_name], "completed", true, allocator);
    } catch (const std::exception& e) {
        add_or_set_member(test_case["spmv"][format_name], "completed", false,
                          allocator);
        std::cerr << "Error when processing test case " << test_case << "\n"
                  << "what(): " << e.what() << std::endl;
    }
}

// This function supposes that management of `FLAGS_overwrite` is done before
// calling it
//
// Overload with the reading a batch matrix with std::vector<matrix_data>
void apply_spmv(const char* format_name, std::shared_ptr<gko::Executor> exec,
                const std::vector<gko::matrix_data<etype>>& data,
                const batch_vec<etype>* b, const batch_vec<etype>* x,
                const batch_vec<etype>* answer, rapidjson::Value& test_case,
                rapidjson::MemoryPoolAllocator<>& allocator)
{
    try {
        auto& spmv_case = test_case["spmv"];
        add_or_set_member(spmv_case, format_name,
                          rapidjson::Value(rapidjson::kObjectType), allocator);

        auto n_dup = FLAGS_num_duplications;
        auto storage_logger = std::make_shared<StorageLogger>();
        exec->add_logger(storage_logger);
        auto system_matrix = share(
            formats::batch_matrix_factory2.at(format_name)(exec, n_dup, data));

        std::clog << "Batch Matrix has: "
                  << system_matrix->get_num_batch_entries()
                  << " batches, each of size ("
                  << system_matrix->get_size().at(0)[0] << ", "
                  << system_matrix->get_size().at(0)[1] << ") " << std::endl;

        exec->remove_logger(gko::lend(storage_logger));
        storage_logger->write_data(spmv_case[format_name], allocator);
        // warm run
        for (unsigned int i = 0; i < FLAGS_warmup; i++) {
            auto x_clone = clone(x);
            exec->synchronize();
            system_matrix->apply(lend(b), lend(x_clone));
            exec->synchronize();
        }

        IterationControl ic{get_timer(exec, FLAGS_gpu_timer)};

        // tuning run
#ifdef GINKGO_BENCHMARK_ENABLE_TUNING
        auto& format_case = spmv_case[format_name];
        if (!format_case.HasMember("tuning")) {
            format_case.AddMember(
                "tuning", rapidjson::Value(rapidjson::kObjectType), allocator);
        }
        auto& tuning_case = format_case["tuning"];
        add_or_set_member(tuning_case, "time",
                          rapidjson::Value(rapidjson::kArrayType), allocator);
        add_or_set_member(tuning_case, "values",
                          rapidjson::Value(rapidjson::kArrayType), allocator);

        // Enable tuning for this portion of code
        gko::_tuning_flag = true;
        // Select some values we want to tune.
        std::vector<gko::size_type> tuning_values{
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
        for (auto val : tuning_values) {
            // Actually set the value that will be tuned. See
            // cuda/components/format_conversion.cuh for an example of how this
            // variable is used.
            gko::_tuned_value = val;
            auto tuning_timer = get_timer(exec, FLAGS_gpu_timer);
            for (auto status : ic.run(false)) {
                auto x_clone = clone(x);
                exec->synchronize();
                tuning_timer->tic();
                system_matrix->apply(lend(b), lend(x_clone));
                tuning_timer->toc();
            }
            tuning_case["time"].PushBack(tuning_timer->compute_average_time(),
                                         allocator);
            tuning_case["values"].PushBack(val, allocator);
        }
        // We put back the flag to false to use the default (non-tuned) values
        // for the following
        gko::_tuning_flag = false;
#endif  // GINKGO_BENCHMARK_ENABLE_TUNING

        // timed run
        auto timer = get_timer(exec, FLAGS_gpu_timer);
        for (auto status : ic.run(false)) {
            auto x_clone = clone(x);
            exec->synchronize();
            timer->tic();
            system_matrix->apply(lend(b), lend(x_clone));
            timer->toc();
        }
        add_or_set_member(spmv_case[format_name], "num_batch_entries",
                          system_matrix->get_num_batch_entries(), allocator);
        add_or_set_member(spmv_case[format_name], "time",
                          timer->compute_average_time(), allocator);

        // compute and write benchmark data
        add_or_set_member(spmv_case[format_name], "completed", true, allocator);
    } catch (const std::exception& e) {
        add_or_set_member(test_case["spmv"][format_name], "completed", false,
                          allocator);
        std::cerr << "Error when processing test case " << test_case << "\n"
                  << "what(): " << e.what() << std::endl;
    }
}


int main(int argc, char* argv[])
{
    std::string header =
        "A benchmark for measuring performance of Ginkgo's spmv.\n";
    std::string format = std::string() + "  [\n" +
                         "    { \"problem\": \"my_file.mtx\"},\n" +
                         "    { \"problem\": \"my_file2.mtx\"}\n" + "  ]\n\n";
    initialize_argument_parsing(&argc, &argv, header, format);

    std::string extra_information = "The formats are " + FLAGS_formats +
                                    "\nThe number of right hand sides is " +
                                    std::to_string(FLAGS_nrhs) + "\n";
    print_general_information(extra_information);

    auto exec = executor_factory.at(FLAGS_executor)(FLAGS_gpu_timer);
    auto engine = get_engine();
    auto formats = split(FLAGS_formats, ',');

    rapidjson::IStreamWrapper jcin(std::cin);
    rapidjson::Document test_cases;
    test_cases.ParseStream(jcin);
    if (!test_cases.IsArray()) {
        if (FLAGS_using_suite_sparse) {
            print_batch_config_error_and_exit();
        } else {
            print_config_error_and_exit();
        }
    }

    auto& allocator = test_cases.GetAllocator();

    for (auto& test_case : test_cases.GetArray()) {
        try {
            // set up benchmark
            if (FLAGS_using_suite_sparse) {
                validate_option_object(test_case);
            } else {
                validate_batch_option_object(test_case);
            }
            if (!test_case.HasMember("spmv")) {
                test_case.AddMember("spmv",
                                    rapidjson::Value(rapidjson::kObjectType),
                                    allocator);
            }
            auto& spmv_case = test_case["spmv"];
            if (!FLAGS_overwrite &&
                all_of(begin(formats), end(formats),
                       [&spmv_case](const std::string& s) {
                           return spmv_case.HasMember(s.c_str());
                       })) {
                continue;
            }
            std::clog << "Running test case: " << test_case << std::endl;
            auto nrhs = FLAGS_nrhs;
            auto nbatch = FLAGS_num_batches;
            auto ndup = FLAGS_num_duplications;
            size_type multiplier = 1;
            auto data = std::vector<gko::matrix_data<etype>>(nbatch);
            auto scale_data = std::vector<gko::matrix_data<etype>>(nbatch);
            if (FLAGS_using_suite_sparse) {
                GKO_ASSERT(ndup == nbatch);
                std::string fname = test_case["filename"].GetString();
                std::ifstream mtx_fd(fname);
                data[0] = gko::read_raw<etype>(mtx_fd);
                multiplier = 1;
            } else {
                for (size_type i = 0; i < data.size(); ++i) {
                    std::string mat_str;
                    if (FLAGS_batch_scaling == "implicit") {
                        mat_str = "A_scaled.mtx";
                    } else {
                        mat_str = "A.mtx";
                    }
                    std::string fbase =
                        std::string(test_case["problem"].GetString()) + "/" +
                        std::to_string(i) + "/";
                    std::string fname = fbase + mat_str;
                    std::ifstream mtx_fd(fname);
                    data[i] = gko::read_raw<etype>(mtx_fd);
                    if (FLAGS_batch_scaling == "explicit") {
                        std::string scale_fname = fbase + "S.mtx";
                        std::ifstream scale_fd(scale_fname);
                        scale_data[i] = gko::read_raw<etype>(scale_fd);
                    }
                }
                multiplier = ndup;
            }

            auto b = batch_vec<etype>::create(exec);
            if (FLAGS_using_suite_sparse) {
                b = create_batch_matrix<etype>(
                    exec,
                    gko::batch_dim<2>(nbatch * multiplier,
                                      gko::dim<2>{data[0].size[0], nrhs}),
                    engine);
            } else {
                if (FLAGS_rhs_generation == "file") {
                    std::vector<gko::matrix_data<etype>> bdata(nbatch);
                    for (size_type i = 0; i < bdata.size(); ++i) {
                        std::string b_str;
                        if (FLAGS_batch_scaling == "implicit") {
                            b_str = "b_scaled.mtx";
                        } else {
                            b_str = "b.mtx";
                        }
                        std::string fname =
                            std::string(test_case["problem"].GetString()) +
                            "/" + std::to_string(i) + "/" + b_str;
                        std::ifstream b_fd(fname);
                        bdata[i] = gko::read_raw<etype>(b_fd);
                    }
                    auto temp_b = batch_vec<etype>::create(exec);
                    temp_b->read(bdata);
                    b = batch_vec<etype>::create(exec, ndup, temp_b.get());
                } else {
                    b = create_batch_matrix<etype>(
                        exec,
                        gko::batch_dim<2>(nbatch * multiplier,
                                          gko::dim<2>{data[0].size[0], nrhs}),
                        engine);
                }
            }
            auto x = create_batch_matrix<etype>(
                exec,
                gko::batch_dim<2>(nbatch * multiplier,
                                  gko::dim<2>{data[0].size[0], nrhs}),
                engine);

            // Compute the result from ginkgo::batch_csr as the correct answer
            auto answer = batch_vec<etype>::create(exec);
            if (FLAGS_detailed) {
                std::shared_ptr<gko::BatchLinOp> system_matrix;
                if (FLAGS_using_suite_sparse) {
                    system_matrix = formats::batch_matrix_factory.at(
                        "batch_csr")(exec, ndup, data[0]);
                } else {
                    system_matrix = formats::batch_matrix_factory2.at(
                        "batch_csr")(exec, ndup, data);
                }
                auto sys_size = system_matrix->get_size();
                auto x_size = x->get_size();
                auto b_size = b->get_size();
                answer->copy_from(lend(x));
                exec->synchronize();
                system_matrix->apply(lend(b), lend(answer));
                exec->synchronize();
            }
            for (const auto& format_name : formats) {
                if (FLAGS_using_suite_sparse) {
                    apply_spmv(format_name.c_str(), exec, data[0], lend(b),
                               lend(x), lend(answer), test_case, allocator);
                } else {
                    apply_spmv(format_name.c_str(), exec, data, lend(b),
                               lend(x), lend(answer), test_case, allocator);
                }
                std::clog << "Current state:" << std::endl
                          << test_cases << std::endl;
                backup_results(test_cases);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error setting up matrix data, what(): " << e.what()
                      << std::endl;
        }
    }

    std::cout << test_cases << std::endl;
}
