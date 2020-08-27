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
#include "third_party/benchmark_wrappers.hpp"

using etype = double;
#ifdef GKO_SPGEMM_LONG
using itype = gko::int64;
const auto benchmark_name = "spgemm64";
#else
using itype = gko::int32;
const auto benchmark_name = "spgemm";
#endif
using Mtx = gko::matrix::Csr<etype, itype>;
using mat_data = gko::matrix_data<etype, itype>;

const std::map<std::string,
               const std::function<std::shared_ptr<gko::LinOp>(
                   std::shared_ptr<gko::Executor>, const mat_data &)>>
    format_map{
        {"multipass",
         [](std::shared_ptr<gko::Executor> exec, const mat_data &data) {
             // prevent double-free on executors/sparselib handles
             std::shared_ptr<Mtx::strategy_type> strategy =
                 dynamic_cast<gko::HipExecutor *>(exec.get())
                     ? std::make_shared<Mtx::load_balance>(
                           gko::as<gko::HipExecutor>(exec))
                     : (dynamic_cast<gko::CudaExecutor *>(exec.get())
                            ? std::make_shared<Mtx::load_balance>(
                                  gko::as<gko::CudaExecutor>(exec))
                            : std::make_shared<Mtx::load_balance>());
             auto mtx =
                 Mtx::create(exec, data.size, data.nonzeros.size(), strategy);
             mtx->read(data);
             return gko::share(mtx);
         }},
        {"twopass",
         [](std::shared_ptr<gko::Executor> exec, const mat_data &data) {
             auto mtx = Mtx::create(exec, data.size, data.nonzeros.size(),
                                    std::make_shared<Mtx::classical>());
             mtx->read(data);
             return gko::share(mtx);
         }},
#ifdef GKO_SPGEMM_HAS_NSPARSE
        {"nsparse",
         [](std::shared_ptr<gko::Executor> exec, const mat_data &data) {
             auto mtx = gko::NSparseCsr<etype>::create(exec, data.size);
             mtx->read(data);
             return gko::share(mtx);
         }},
#endif
#ifdef GKO_SPGEMM_HAS_ACSPGEMM
        {"acspgemm",
         [](std::shared_ptr<gko::Executor> exec, const mat_data &data) {
             auto mtx = gko::AcCsr<etype>::create(exec, data.size);
             mtx->read(data);
             return gko::share(mtx);
         }},
#endif
#ifdef GKO_SPGEMM_HAS_SPECK
        {"speck",
         [](std::shared_ptr<gko::Executor> exec, const mat_data &data) {
             auto mtx = gko::SpeckCsr<etype>::create(exec, data.size);
             mtx->read(data);
             return gko::share(mtx);
         }},
#endif
#ifdef GKO_SPGEMM_HAS_KOKKOS
        {"kokkos",
         [](std::shared_ptr<gko::Executor> exec, const mat_data &data) {
             auto mtx = gko::KokkosCsr<etype>::create(exec, data.size);
             mtx->read(data);
             return gko::share(mtx);
         }},
#endif
        {"sparselib",
         [](std::shared_ptr<gko::Executor> exec, const mat_data &data) {
             auto mtx = Mtx::create(exec, data.size, data.nonzeros.size(),
                                    std::make_shared<Mtx::sparselib>());
             mtx->read(data);
             return gko::share(mtx);
         }}};


DEFINE_int32(rowlength, 10,
             "The length of rows in randomly generated matrices B. Only "
             "relevant for mode = <sparse|dense>");


std::shared_ptr<Mtx> get_csr_base(std::shared_ptr<gko::LinOp> op)
{
#ifdef GKO_SPGEMM_HAS_NSPARSE
    if (dynamic_cast<gko::NSparseCsr<etype> *>(op.get())) {
        return gko::as<gko::NSparseCsr<etype>>(op)->get_matrix();
    }
#endif
#ifdef GKO_SPGEMM_HAS_ACSPGEMM
    if (dynamic_cast<gko::AcCsr<etype> *>(op.get())) {
        return gko::as<gko::AcCsr<etype>>(op)->get_matrix();
    }
#endif
#ifdef GKO_SPGEMM_HAS_SPECK
    if (dynamic_cast<gko::SpeckCsr<etype> *>(op.get())) {
        return gko::as<gko::SpeckCsr<etype>>(op)->get_matrix();
    }
#endif
#ifdef GKO_SPGEMM_HAS_KOKKOS
    if (dynamic_cast<gko::KokkosCsr<etype> *>(op.get())) {
        return gko::as<gko::KokkosCsr<etype>>(op)->get_matrix();
    }
#endif
    return gko::as<Mtx>(op);
}


const std::map<std::string, const std::function<std::shared_ptr<Mtx>(
                                std::shared_ptr<gko::LinOp>)>>
    mode_map{
        {"normal",
         [](std::shared_ptr<gko::LinOp> matrix) {
             return matrix->get_size()[0] == matrix->get_size()[1]
                        ? get_csr_base(matrix)
                        : gko::as<Mtx>(get_csr_base(matrix)->transpose());
         }},
        {"transposed",
         [](std::shared_ptr<gko::LinOp> matrix) {
             return gko::as<Mtx>(get_csr_base(matrix)->transpose());
         }},
        {"sparse",
         [](std::shared_ptr<gko::LinOp> matrix) {
             auto size = gko::transpose(matrix->get_size());
             // don't expect too much quality from this seed =)
             std::default_random_engine rng(
                 FLAGS_seed ^ (matrix->get_size()[0] << 24) ^
                 (matrix->get_size()[1] << 15) -
                     get_csr_base(matrix)->get_num_stored_elements());
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
             data.ensure_row_major_order();
             auto mtx = Mtx::create(matrix->get_executor(), size);
             mtx->read(data);
             return gko::share(std::move(mtx));
         }},
        {"dense", [](std::shared_ptr<gko::LinOp> matrix) {
             auto size = gko::dim<2>(matrix->get_size()[1], FLAGS_rowlength);
             // don't expect too much quality from this seed =)
             std::default_random_engine rng(
                 FLAGS_seed ^ (matrix->get_size()[0] << 24) ^
                 (matrix->get_size()[1] << 15) -
                     get_csr_base(matrix)->get_num_stored_elements());
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
    strategies,
    "multipass,twopass,sparselib"
#ifdef GKO_SPGEMM_HAS_NSPARSE
    ",nsparse"
#endif
#ifdef GKO_SPGEMM_HAS_ACSPGEMM
    ",acspgemm"
#endif
#ifdef GKO_SPGEMM_HAS_SPECK
    ",speck"
#endif
#ifdef GKO_SPGEMM_HAS_KOKKOS
    ",kokkos"
#endif
    ,
    "Comma-separated list of SpGEMM strategies: multipass, twopass, sparselib"
#ifdef GKO_SPGEMM_HAS_NSPARSE
    ", nsparse"
#endif
#ifdef GKO_SPGEMM_HAS_ACSPGEMM
    ", acspgemm"
#endif
#ifdef GKO_SPGEMM_HAS_SPECK
    ", speck"
#endif
#ifdef GKO_SPGEMM_HAS_KOKKOS
    ", kokkos"
#endif
);


DEFINE_bool(compute_work, false, "Compute FLOP and nnz count of the SpGEMM");


std::pair<gko::int64, gko::int64> compute_spgemm_work_and_nnz(
    const gko::matrix_data<etype, itype> &data)
{
    auto ref = gko::ReferenceExecutor::create();
    auto ref_mtx = gko::share(Mtx::create(ref));
    ref_mtx->read(data);
    auto ref_mtx2 = mode_map.at(FLAGS_mode)(ref_mtx);

    auto num_rows = ref_mtx->get_size()[0];
    gko::int64 total_count{};
    gko::int64 nnz{};
    std::unordered_set<itype> columns;
    // for each row of A ...
    for (gko::size_type row = 0; row < num_rows; ++row) {
        columns.clear();
        auto begin = ref_mtx->get_const_row_ptrs()[row];
        auto end = ref_mtx->get_const_row_ptrs()[row + 1];
        // sum up the size of all corresponding rows of B
        for (auto nz = begin; nz < end; ++nz) {
            auto col = ref_mtx->get_const_col_idxs()[nz];
            auto b_begin = ref_mtx2->get_const_row_ptrs()[col];
            auto b_end = ref_mtx2->get_const_row_ptrs()[col + 1];
            total_count += b_end - b_begin;
            columns.insert(ref_mtx2->get_const_col_idxs() + b_begin,
                           ref_mtx2->get_const_col_idxs() + b_end);
        }
        nnz += columns.size();
    }
    return {total_count, nnz};
}


std::shared_ptr<Mtx> compute_spgemm_ref(
    const gko::matrix_data<etype, itype> &data)
{
    auto ref = gko::ReferenceExecutor::create();
    auto ref_mtx = gko::share(Mtx::create(ref));
    ref_mtx->read(data);
    auto ref_mtx2 = mode_map.at(FLAGS_mode)(ref_mtx);

    auto ref_res = Mtx::create(
        ref, gko::dim<2>(ref_mtx->get_size()[0], ref_mtx2->get_size()[1]));

    ref_mtx->apply(gko::lend(ref_mtx2), gko::lend(ref_res));
    return gko::share(ref_res);
}

std::pair<bool, double> validate_spgemm(const Mtx *reference_solution,
                                        const Mtx *device_result)
{
    auto ref = gko::ReferenceExecutor::create();
    auto result = Mtx::create(ref);
    result->copy_from(gko::lend(device_result));
    if (reference_solution->get_num_stored_elements() !=
        result->get_num_stored_elements()) {
        return {false, 0.0};
    }

    auto num_rows = reference_solution->get_size()[0];
    double error{};
    // for each row
    for (gko::size_type row = 0; row < num_rows; ++row) {
        // check for equal row lenghts
        auto begin = reference_solution->get_const_row_ptrs()[row];
        auto end = reference_solution->get_const_row_ptrs()[row + 1];
        auto begin2 = result->get_const_row_ptrs()[row];
        auto end2 = result->get_const_row_ptrs()[row + 1];
        if (begin != begin2 || end != end2) {
            return {false, 0.0};
        }
        // for each non-zero
        for (auto nz = begin; nz < end; ++nz) {
            // check for equal column indices
            auto col = reference_solution->get_const_col_idxs()[nz];
            auto col2 = result->get_const_col_idxs()[nz];
            if (col != col2) {
                return {false, 0.0};
            }
            // compute value error
            auto val = reference_solution->get_const_values()[nz];
            auto val2 = result->get_const_values()[nz];
            error += gko::squared_norm(val - val2);
        }
    }
    return {true, std::sqrt(error)};
}

void apply_spgemm(const char *strategy_name,
                  std::shared_ptr<gko::Executor> exec,
                  const gko::matrix_data<etype, itype> &data,
                  std::shared_ptr<Mtx> reference_solution,
                  rapidjson::Value &test_case,
                  rapidjson::MemoryPoolAllocator<> &allocator)
{
    try {
        add_or_set_member(test_case, strategy_name,
                          rapidjson::Value(rapidjson::kObjectType), allocator);
        add_or_set_member(test_case[strategy_name], "mode",
                          rapidjson::Value(FLAGS_mode.c_str(), allocator),
                          allocator);

        auto mtx = format_map.at(strategy_name)(exec, data);
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
        auto validation_result =
            validate_spgemm(gko::lend(reference_solution), gko::lend(res));
        add_or_set_member(test_case[strategy_name], "correct",
                          validation_result.first, allocator);
        add_or_set_member(test_case[strategy_name], "error",
                          validation_result.second, allocator);
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
            auto &spgemm_case = test_case[benchmark_name];
            if (!FLAGS_overwrite &&
                all_of(begin(strategies), end(strategies),
                       [&](const std::string &s) {
                           return spgemm_case.HasMember(s.c_str());
                       })) {
                continue;
            }
            std::clog << "Running test case: " << test_case << std::endl;
            std::ifstream mtx_fd(test_case["filename"].GetString());
            auto data = gko::read_raw<etype, itype>(mtx_fd);
            data.ensure_row_major_order();

            // compute the exact amount of products a_ik * b_kj the SpGEMM has
            // to compute
            gko::int64 total_work{};
            gko::int64 total_nnz{};
            if (FLAGS_compute_work) {
                auto spgemm_stats = compute_spgemm_work_and_nnz(data);
                total_work = spgemm_stats.first;
                total_nnz = spgemm_stats.second;

                // store the amount of work and output nnz for SpGEMM
                add_or_set_member(test_case, "spgemm_work", total_work,
                                  allocator);
                add_or_set_member(test_case, "spgemm_nnz", total_nnz,
                                  allocator);
            }

            std::clog << "Matrix is of size (" << data.size[0] << ", "
                      << data.size[1] << "), " << data.nonzeros.size() << ", "
                      << total_work << ", " << total_nnz << std::endl;

            if (total_nnz > std::numeric_limits<itype>::max()) {
                std::clog << "Computing the product A*B would overflow the "
                             "index type, skipping"
                          << std::endl;
                add_or_set_member(test_case, "skipped", true, allocator);
                backup_results(test_cases);
                continue;
            }

            // compute reference solution
            auto reference_solution = compute_spgemm_ref(data);

            for (const auto &strategy_name : strategies) {
                apply_spgemm(strategy_name.c_str(), exec, data,
                             reference_solution, spgemm_case, allocator);
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
