// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <ginkgo/ginkgo.hpp>


#define GKO_BENCHMARK_DISTRIBUTED


#include "benchmark/utils/general_matrix.hpp"
#include "benchmark/utils/generator.hpp"
#include "benchmark/utils/iteration_control.hpp"
#include "benchmark/utils/runner.hpp"
#include "benchmark/utils/structure.hpp"
#include "benchmark/utils/timer.hpp"
#include "benchmark/utils/types.hpp"


DEFINE_string(operations, "spgemm",
              "Comma-separated list of operations to be benchmarked. Can be "
              "spgemm, transpose");

DEFINE_string(spgemm_mode, "normal",
              R"(Which matrix B should be used to compute A * B: normal,
transposed, sparse
normal: B = A for A square, A^T otherwise
transposed: B = A^T
sparse: B is a sparse matrix with dimensions of A^T with uniformly
        random values, at most -spgemm_rowlength non-zeros per row)");

DEFINE_int32(spgemm_rowlength, 10,
             "The length of rows in randomly generated matrices B. Only "
             "relevant for spgemm_mode = sparse");


using Generator = DistributedDefaultSystemGenerator<DefaultSystemGenerator<>>;
using DistMtx = Generator::Mtx;


struct DistributedSparseBlasBenchmark : Benchmark<std::shared_ptr<DistMtx>> {
    std::string name;
    std::vector<std::string> operations;
    Generator generator;
    bool do_print;

    DistributedSparseBlasBenchmark(Generator generator, bool do_print)
        : name{"sparse_blas"},
          operations{split(FLAGS_operations)},
          generator{std::move(generator)},
          do_print{do_print}
    {}

    const std::string& get_name() const override { return name; }

    const std::vector<std::string>& get_operations() const override
    {
        return operations;
    }

    bool should_print() const override { return do_print; }

    bool validate_config(const json& value) const override
    {
        return generator.validate_config(value);
    }

    std::string get_example_config() const override
    {
        return generator.get_example_config();
    }

    std::string describe_config(const json& test_case) const override
    {
        return generator.describe_config(test_case);
    }

    std::shared_ptr<DistMtx> setup(std::shared_ptr<gko::Executor> exec,
                                   json& test_case) const override
    {
        auto [data, local_size] = generator.generate_matrix_data(test_case);
        reorder(data, test_case);

        // the global count: a rank's data has only its own rows
        gko::int64 nonzeros = data.nonzeros.size();
        generator.comm.all_reduce(exec->get_master(), &nonzeros, 1, MPI_SUM);
        if (do_print) {
            std::cerr << "Matrix is of size (" << data.size[0] << ", "
                      << data.size[1] << "), " << nonzeros << std::endl;
        }
        test_case["rows"] = data.size[0];
        test_case["cols"] = data.size[1];
        test_case["nonzeros"] = nonzeros;

        // multiply needs Csr blocks
        return gko::as<DistMtx>(generator.generate_matrix_with_format(
            exec, "csr-csr", data, local_size));
    }

    // A^T, filled by the collective transpose: a valid right operand of A
    std::shared_ptr<DistMtx> transposed(std::shared_ptr<gko::Executor> exec,
                                        const DistMtx* mtx) const
    {
        auto result = gko::share(DistMtx::create(exec, generator.comm));
        mtx->transpose(result);
        return result;
    }

    // A matrix of the size of A^T with min(rowlength, columns) distinct
    // random columns in each row. Each rank fills its own rows, from a
    // generator seeded by -seed and the rank. mt19937_64 and Floyd's sampling
    // give the same pattern with every standard library, so the nonzeros of
    // the product are reproducible.
    std::shared_ptr<DistMtx> random_sparse(std::shared_ptr<gko::Executor> exec,
                                           const DistMtx* mtx) const
    {
        // the generator reads A with one partition for its rows and columns
        auto part = mtx->get_row_partition();
        auto host_part = gko::clone(exec->get_master(), part);
        const auto size = gko::transpose(mtx->get_size());
        const auto cols = static_cast<global_itype>(size[1]);
        const auto rowlength =
            std::min<global_itype>(std::max(FLAGS_spgemm_rowlength, 0), cols);
        const auto rank = generator.comm.rank();
        std::mt19937_64 engine(FLAGS_seed + rank);
        std::uniform_real_distribution<gko::remove_complex<etype>> val_dist(
            -1.0, 1.0);
        gko::matrix_data<etype, global_itype> data{size};
        std::unordered_set<global_itype> picked;
        for (gko::size_type range = 0; range < host_part->get_num_ranges();
             ++range) {
            if (host_part->get_part_ids()[range] != rank) {
                continue;
            }
            for (auto row = host_part->get_range_bounds()[range];
                 row < host_part->get_range_bounds()[range + 1]; ++row) {
                picked.clear();
                for (auto j = cols - rowlength; j < cols; ++j) {
                    const auto t = static_cast<global_itype>(
                        engine() % static_cast<gko::uint64>(j + 1));
                    picked.insert(picked.count(t) ? j : t);
                }
                for (auto col : picked) {
                    data.nonzeros.emplace_back(
                        row, col,
                        gko::detail::get_rand_value<etype>(val_dist, engine));
                }
            }
        }
        data.sort_row_major();
        auto result = gko::share(DistMtx::create(exec, generator.comm));
        result->read_distributed(data, part, part);
        return result;
    }

    // B of C = A * B, by -spgemm_mode
    std::shared_ptr<const DistMtx> spgemm_operand(
        std::shared_ptr<gko::Executor> exec,
        const std::shared_ptr<DistMtx>& mtx) const
    {
        const std::string mode{FLAGS_spgemm_mode};
        const auto size = mtx->get_size();
        if (mode == "normal") {
            if (size[0] == size[1]) {
                return mtx;
            }
            return transposed(exec, mtx.get());
        } else if (mode == "transposed") {
            return transposed(exec, mtx.get());
        } else if (mode == "sparse") {
            return random_sparse(exec, mtx.get());
        }
        throw gko::Error{
            __FILE__, __LINE__,
            "Unsupported SpGEMM mode " + mode + " for distributed matrices"};
    }

    void run(std::shared_ptr<gko::Executor> exec, std::shared_ptr<Timer> timer,
             annotate_functor annotate, std::shared_ptr<DistMtx>& mtx,
             const std::string& operation_name,
             json& operation_case) const override
    {
        auto result = DistMtx::create(exec, generator.comm);
        std::function<void()> op;
        std::shared_ptr<const DistMtx> b;
        if (operation_name == "spgemm") {
            b = spgemm_operand(exec, mtx);
            op = [&] { mtx->multiply(b, result); };
        } else if (operation_name == "transpose") {
            op = [&] { mtx->transpose(result); };
        } else {
            throw std::runtime_error("Operation " + operation_name +
                                     " is not supported for distributed "
                                     "matrices, only spgemm and transpose");
        }
        IterationControl ic(timer);

        {
            auto range = annotate("warmup", FLAGS_warmup > 0);
            for (auto _ : ic.warmup_run()) {
                op();
            }
        }

        for (auto _ : ic.run()) {
            auto range = annotate("repetition");
            op();
        }

        write_times(operation_case, ic);
        operation_case["nonzeros_out"] = describe_operator(result.get()).second;
    }
};


int main(int argc, char* argv[])
{
    gko::experimental::mpi::environment mpi_env{argc, argv};

    const auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    const auto do_print = comm.rank() == 0;

    std::string header =
        "A benchmark for measuring the performance of Ginkgo's distributed "
        "sparse BLAS operations.\n";
    std::string format = Generator::get_example_config();
    initialize_argument_parsing_matrix(&argc, &argv, header, format, "",
                                       do_print);

    auto exec = executor_factory_mpi.at(FLAGS_executor)(comm.get());

    if (do_print) {
        print_general_information("The operations are " + FLAGS_operations,
                                  exec);
    }

    std::string json_input = broadcast_json_input(get_input_stream(), comm);
    auto test_cases = json::parse(json_input);

    run_test_cases(DistributedSparseBlasBenchmark{Generator{comm}, do_print},
                   exec, get_mpi_timer(exec, comm, FLAGS_gpu_timer),
                   test_cases);

    if (do_print) {
        std::cout << std::setw(4) << test_cases << std::endl;
    }
}
