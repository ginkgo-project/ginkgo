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
#include "benchmark/utils/types.hpp"
#include "core/test/utils/matrix_generator.hpp"


const auto benchmark_name = "sparse_blas";


using itype = gko::int32;
using Mtx = gko::matrix::Csr<etype, itype>;
using mat_data = gko::matrix_data<etype, itype>;


const std::map<std::string,
               const std::function<std::shared_ptr<Mtx::strategy_type>()>>
    strategy_map{
        {"classical", [] { return std::make_shared<Mtx::classical>(); }},
        {"sparselib", [] { return std::make_shared<Mtx::sparselib>(); }}};

DEFINE_string(operations, "spgemm,spgeam,transpose",
              "Comma-separated list of operations to be benchmarked. Can be "
              "spgemm, spgeam, transpose");

DEFINE_string(strategies, "classical,sparselib",
              "Comma-separated list of CSR strategies: classical, sparselib");

DEFINE_int32(
    spgeam_swap_distance, 100,
    "Maximum distance for row swaps to avoid rows with disjoint column ranges");

DEFINE_string(
    spgemm_mode, "normal",
    "Which matrix B should be used to compute A * B: normal, "
    "transposed, sparse, dense\n"
    "normal: B = A for A square, A^T otherwise\ntransposed: B = "
    "A^T\nsparse: B is a sparse matrix with dimensions of A^T with uniformly "
    "random values, at most -spgemm_rowlength non-zeros per row\ndense: B is a "
    "'dense' sparse matrix with -spgemm_rowlength columns and non-zeros per "
    "row");

DEFINE_int32(spgemm_rowlength, 10,
             "The length of rows in randomly generated matrices B. Only "
             "relevant for spgemm_mode = <sparse|dense>");

DEFINE_bool(validate_results, false,
            "Check for correct sparsity pattern and compute the L2 norm "
            "against the ReferenceExecutor solution.");


std::pair<bool, double> validate_result(const Mtx* correct_mtx,
                                        const Mtx* host_mtx)
{
    if (correct_mtx->get_size() != host_mtx->get_size() ||
        correct_mtx->get_num_stored_elements() !=
            host_mtx->get_num_stored_elements()) {
        return {false, 0.0};
    }
    double err_nrm_sq{};
    const auto size = correct_mtx->get_size();
    for (gko::size_type row = 0; row < size[0]; row++) {
        const auto begin = host_mtx->get_const_row_ptrs()[row];
        const auto end = host_mtx->get_const_row_ptrs()[row + 1];
        if (begin != correct_mtx->get_const_row_ptrs()[row] ||
            end != correct_mtx->get_const_row_ptrs()[row + 1] ||
            !std::equal(correct_mtx->get_const_col_idxs() + begin,
                        correct_mtx->get_const_col_idxs() + end,
                        host_mtx->get_const_col_idxs() + begin)) {
            return {false, 0.0};
        }
        for (auto nz = begin; nz < end; nz++) {
            const auto diff = host_mtx->get_const_values()[nz] -
                              correct_mtx->get_const_values()[nz];
            err_nrm_sq += gko::squared_norm(diff);
        }
    }
    return {true, sqrt(err_nrm_sq)};
}


class BenchmarkOperation {
public:
    virtual ~BenchmarkOperation() = default;

    /**
     * Computes an estimate for the number of FLOPs executed by the operation.
     */
    virtual gko::size_type get_flops() const = 0;

    /**
     * Computes an estimate for the amount of memory accessed by the operation
     * (bytes).
     */
    virtual gko::size_type get_memory() const = 0;

    /**
     * Sets up all necessary data for a following call to
     * BenchmarkOperation::run.
     */
    virtual void prepare(){};

    /**
     * Computes the error between a reference solution and the solution provided
     * by this operation. The first value specifies whether the result is
     * structurally correct, the second value specifies the numerical error.
     */
    virtual std::pair<bool, double> validate() const = 0;

    /**
     * Executes the operation to be benchmarked.
     */
    virtual void run() = 0;
};


class SpgemmOperation : public BenchmarkOperation {
public:
    explicit SpgemmOperation(const Mtx* mtx) : mtx_{mtx}
    {
        auto exec = mtx_->get_executor();
        const auto size = mtx_->get_size();
        std::string mode_str{FLAGS_spgemm_mode};
        if (mode_str == "normal") {
            // normal for square matrix, transposed for rectangular
            if (size[0] == size[1]) {
                mtx2_ = mtx_->clone();
            } else {
                mtx2_ = gko::as<Mtx>(mtx_->transpose());
            }
        } else if (mode_str == "transposed") {
            // always transpose
            mtx2_ = gko::as<Mtx>(mtx_->transpose());
        } else if (mode_str == "sparse") {
            // create sparse matrix of transposed size
            const auto size2 = gko::transpose(size);
            std::default_random_engine rng(FLAGS_seed);
            std::uniform_real_distribution<gko::remove_complex<etype>> val_dist(
                -1.0, 1.0);
            gko::matrix_data<etype, itype> data{size, {}};
            const auto local_rowlength =
                std::min<int>(FLAGS_spgemm_rowlength, size2[1]);
            data.nonzeros.reserve(size2[0] * local_rowlength);
            // randomly permute column indices
            std::vector<itype> cols(size2[1]);
            std::iota(cols.begin(), cols.end(), 0);
            for (gko::size_type row = 0; row < size2[0]; ++row) {
                std::shuffle(cols.begin(), cols.end(), rng);
                for (int i = 0; i < local_rowlength; ++i) {
                    data.nonzeros.emplace_back(
                        row, cols[i],
                        gko::detail::get_rand_value<etype>(val_dist, rng));
                }
            }
            data.ensure_row_major_order();
            mtx2_ = Mtx::create(exec, size2);
            mtx2_->read(data);
        } else if (mode_str == "dense") {
            const auto size2 = gko::dim<2>(size[1], FLAGS_spgemm_rowlength);
            // don't expect too much quality from this seed =)
            std::default_random_engine rng(FLAGS_seed);
            std::uniform_real_distribution<gko::remove_complex<etype>> dist(
                -1.0, 1.0);
            gko::matrix_data<etype, itype> data{size2, dist, rng};
            data.ensure_row_major_order();
            mtx2_ = Mtx::create(exec, size2);
            mtx2_->read(data);
        } else {
            throw gko::Error{__FILE__, __LINE__,
                             "Unsupported SpGEMM mode " + mode_str};
        }
    }

    std::pair<bool, double> validate() const override
    {
        auto ref = gko::ReferenceExecutor::create();
        auto correct = Mtx::create(ref, mtx_out_->get_size());
        gko::clone(ref, mtx_)->apply(mtx2_.get(), correct.get());
        return validate_result(correct.get(), gko::clone(ref, mtx_out_).get());
    }

    gko::size_type get_flops() const override
    {
        auto host_mtx = Mtx::create(mtx_->get_executor());
        auto host_mtx2 = Mtx::create(mtx_->get_executor());
        host_mtx->copy_from(mtx_);
        host_mtx2->copy_from(mtx2_.get());
        // count the individual products a_ik * b_kj
        gko::size_type work{};
        for (gko::size_type row = 0; row < host_mtx->get_size()[0]; row++) {
            auto begin = host_mtx->get_const_row_ptrs()[row];
            auto end = host_mtx->get_const_row_ptrs()[row + 1];
            for (auto nz = begin; nz < end; nz++) {
                auto col = host_mtx->get_const_col_idxs()[nz];
                auto local_work = host_mtx2->get_const_row_ptrs()[col + 1] -
                                  host_mtx2->get_const_row_ptrs()[col];
                work += local_work;
            }
        }
        return 2 * work;
    }

    gko::size_type get_memory() const override
    {
        // read and write everything only once, ignore row pointers
        return (mtx_->get_num_stored_elements() +
                mtx2_->get_num_stored_elements() +
                mtx_out_->get_num_stored_elements()) *
               (sizeof(etype) + sizeof(itype));
    }

    void prepare() override
    {
        mtx_out_ =
            Mtx::create(mtx_->get_executor(),
                        gko::dim<2>{mtx_->get_size()[0], mtx2_->get_size()[1]});
    }

    void run() override { mtx_->apply(lend(mtx2_), lend(mtx_out_)); }

private:
    const Mtx* mtx_;
    std::unique_ptr<Mtx> mtx2_;
    std::unique_ptr<Mtx> mtx_out_;
};


class SpgeamOperation : public BenchmarkOperation {
public:
    explicit SpgeamOperation(const Mtx* mtx) : mtx_{mtx}
    {
        auto exec = mtx_->get_executor();
        const auto size = mtx_->get_size();
        // randomly permute n/2 rows with limited distances
        gko::array<itype> permutation_array(exec->get_master(), size[0]);
        auto permutation = permutation_array.get_data();
        std::iota(permutation, permutation + size[0], 0);
        std::default_random_engine rng(FLAGS_seed);
        std::uniform_int_distribution<itype> start_dist(0, size[0] - 1);
        std::uniform_int_distribution<itype> delta_dist(
            -FLAGS_spgeam_swap_distance, FLAGS_spgeam_swap_distance);
        for (itype i = 0; i < size[0] / 2; ++i) {
            auto a = start_dist(rng);
            auto b = a + delta_dist(rng);
            if (b >= 0 && b < size[0]) {
                std::swap(permutation[a], permutation[b]);
            }
        }
        mtx2_ = gko::as<Mtx>(mtx_->row_permute(&permutation_array));
        id_ = gko::matrix::Identity<etype>::create(exec, size[1]);
        scalar_ = gko::initialize<gko::matrix::Dense<etype>>({1.0}, exec);
    }

    std::pair<bool, double> validate() const override
    {
        auto ref = gko::ReferenceExecutor::create();
        auto correct = gko::clone(ref, mtx2_.get());
        gko::clone(ref, mtx_)->apply(scalar_.get(), id_.get(), scalar_.get(),
                                     correct.get());
        return validate_result(correct.get(), gko::clone(ref, mtx_out_).get());
    }

    gko::size_type get_flops() const override
    {
        return mtx_->get_num_stored_elements() +
               mtx2_->get_num_stored_elements();
    }

    gko::size_type get_memory() const override
    {
        // read and write everything only once, ignore row pointers
        return (mtx_->get_num_stored_elements() +
                mtx2_->get_num_stored_elements() +
                mtx_out_->get_num_stored_elements()) *
               (sizeof(etype) + sizeof(itype));
    }

    void prepare() override { mtx_out_ = mtx2_->clone(); }

    void run() override
    {
        mtx_->apply(scalar_.get(), id_.get(), scalar_.get(), mtx_out_.get());
    }

private:
    const Mtx* mtx_;
    std::unique_ptr<Mtx> mtx2_;
    std::unique_ptr<gko::matrix::Dense<etype>> scalar_;
    std::unique_ptr<gko::matrix::Identity<etype>> id_;
    std::unique_ptr<Mtx> mtx_out_;
};


class TransposeOperation : public BenchmarkOperation {
public:
    explicit TransposeOperation(const Mtx* mtx) : mtx_{mtx} {}

    std::pair<bool, double> validate() const override
    {
        auto ref = gko::ReferenceExecutor::create();
        return validate_result(
            gko::as<Mtx>(gko::clone(ref, mtx_)->transpose()).get(),
            gko::clone(ref, mtx_out_).get());
    }

    gko::size_type get_flops() const override { return 0; }

    gko::size_type get_memory() const override
    {
        // read and write everything only once, ignore row pointers
        return 2 * mtx_->get_num_stored_elements() *
               (sizeof(etype) + sizeof(itype));
    }

    void prepare() override { mtx_out_ = nullptr; }

    void run() override { mtx_out_ = gko::as<Mtx>(mtx_->transpose()); }

private:
    const Mtx* mtx_;
    std::unique_ptr<Mtx> mtx_out_;
};


const std::map<std::string,
               std::function<std::unique_ptr<BenchmarkOperation>(const Mtx*)>>
    operation_map{
        {"spgemm",
         [](const Mtx* mtx) { return std::make_unique<SpgemmOperation>(mtx); }},
        {"spgeam",
         [](const Mtx* mtx) { return std::make_unique<SpgeamOperation>(mtx); }},
        {"transpose", [](const Mtx* mtx) {
             return std::make_unique<TransposeOperation>(mtx);
         }}};


void apply_sparse_blas(const char* operation_name,
                       std::shared_ptr<gko::Executor> exec, const Mtx* mtx,
                       rapidjson::Value& test_case,
                       rapidjson::MemoryPoolAllocator<>& allocator)
{
    try {
        add_or_set_member(test_case, operation_name,
                          rapidjson::Value(rapidjson::kObjectType), allocator);

        auto op = operation_map.at(operation_name)(mtx);

        auto timer = get_timer(exec, FLAGS_gpu_timer);
        IterationControl ic(timer);

        // warm run
        for (auto _ : ic.warmup_run()) {
            op->prepare();
            exec->synchronize();
            op->run();
            exec->synchronize();
        }

        // timed run
        op->prepare();
        for (auto _ : ic.run()) {
            op->run();
        }
        const auto runtime = ic.compute_average_time();
        const auto flops = static_cast<double>(op->get_flops());
        const auto mem = static_cast<double>(op->get_memory());
        const auto repetitions = ic.get_num_repetitions();
        add_or_set_member(test_case[operation_name], "time", runtime,
                          allocator);
        add_or_set_member(test_case[operation_name], "flops", flops / runtime,
                          allocator);
        add_or_set_member(test_case[operation_name], "bandwidth", mem / runtime,
                          allocator);
        add_or_set_member(test_case[operation_name], "repetitions", repetitions,
                          allocator);

        if (FLAGS_validate_results) {
            auto validation_result = op->validate();
            add_or_set_member(test_case[operation_name], "correct",
                              validation_result.first, allocator);
            add_or_set_member(test_case[operation_name], "error",
                              validation_result.second, allocator);
        }

        add_or_set_member(test_case[operation_name], "completed", true,
                          allocator);
    } catch (const std::exception& e) {
        add_or_set_member(test_case[operation_name], "completed", false,
                          allocator);
        if (FLAGS_keep_errors) {
            rapidjson::Value msg_value;
            msg_value.SetString(e.what(), allocator);
            add_or_set_member(test_case[operation_name], "error", msg_value,
                              allocator);
        }
        std::cerr << "Error when processing test case " << test_case << "\n"
                  << "what(): " << e.what() << std::endl;
    }
}


int main(int argc, char* argv[])
{
    std::string header =
        "A benchmark for measuring performance of Ginkgo's sparse BLAS "
        "operations.\n";
    std::string format = std::string() + "  [\n" +
                         "    { \"filename\": \"my_file.mtx\"},\n" +
                         "    { \"filename\": \"my_file2.mtx\"}\n" + "  ]\n\n";
    initialize_argument_parsing(&argc, &argv, header, format);

    auto exec = executor_factory.at(FLAGS_executor)(FLAGS_gpu_timer);

    rapidjson::IStreamWrapper jcin(std::cin);
    rapidjson::Document test_cases;
    test_cases.ParseStream(jcin);
    if (!test_cases.IsArray()) {
        print_config_error_and_exit();
    }

    std::string extra_information = "The operations are " + FLAGS_operations;
    print_general_information(extra_information);

    auto& allocator = test_cases.GetAllocator();

    auto strategies = split(FLAGS_strategies, ',');
    auto operations = split(FLAGS_operations, ',');

    for (auto& test_case : test_cases.GetArray()) {
        try {
            // set up benchmark
            validate_option_object(test_case);
            if (!test_case.HasMember(benchmark_name)) {
                test_case.AddMember(rapidjson::Value(benchmark_name, allocator),
                                    rapidjson::Value(rapidjson::kObjectType),
                                    allocator);
            }
            auto& sp_blas_case = test_case[benchmark_name];
            std::clog << "Running test case: " << test_case << std::endl;
            std::ifstream mtx_fd(test_case["filename"].GetString());
            auto data = gko::read_generic_raw<etype, itype>(mtx_fd);
            data.ensure_row_major_order();
            std::clog << "Matrix is of size (" << data.size[0] << ", "
                      << data.size[1] << "), " << data.nonzeros.size()
                      << std::endl;

            for (const auto& strategy_name : strategies) {
                auto mtx = Mtx::create(exec, data.size, data.nonzeros.size(),
                                       strategy_map.at(strategy_name)());
                mtx->read(data);
                for (const auto& operation_name : operations) {
                    const auto name = operation_name + "-" + strategy_name;
                    if (FLAGS_overwrite ||
                        !sp_blas_case.HasMember(name.c_str())) {
                        apply_sparse_blas(operation_name.c_str(), exec,
                                          mtx.get(), sp_blas_case, allocator);
                        std::clog << "Current state:" << std::endl
                                  << test_cases << std::endl;
                        backup_results(test_cases);
                    }
                }
            }
            // write the output if we have no strategies
            backup_results(test_cases);
        } catch (const std::exception& e) {
            std::cerr << "Error setting up matrix data, what(): " << e.what()
                      << std::endl;
        }
    }

    std::cout << test_cases << std::endl;
}
