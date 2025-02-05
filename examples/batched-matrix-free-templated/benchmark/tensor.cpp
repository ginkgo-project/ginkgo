// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <benchmark/utils/general.hpp>
#include <benchmark/utils/general_matrix.hpp>
#include <benchmark/utils/iteration_control.hpp>
#include <benchmark/utils/runner.hpp>
#include <core/test/utils/array_generator.hpp>
#include <core/test/utils/batch_helpers.hpp>
#include <examples/batched-matrix-free-templated/tensor_left.hpp>

DEFINE_string(
    apply, "matrix-free",
    "The apply implementation: either >matrix-free<, >matrix-dense<, or "
    ">matrix-sparse<, or a >,< separated list.");

using vtype = tensor::ValueType;


std::string get_example_config()
{
    return json::parse(R"([{"size_1d": 4, "num_batches": 10}])").dump(4);
}

struct TensorState {
    std::unique_ptr<gko::batch::matrix::Dense<vtype>> data_1d;
    std::unique_ptr<gko::batch::MultiVector<vtype>> x;
    std::unique_ptr<gko::batch::MultiVector<vtype>> b;
};

struct TensorBenchmark : public Benchmark<TensorState> {
    std::vector<std::string> operations = split(FLAGS_apply);
    std::string name = "Tensor";

    const std::string& get_name() const override { return name; }
    const std::vector<std::string>& get_operations() const override
    {
        return operations;
    }
    bool should_print() const override { return true; }
    std::string get_example_config() const override
    {
        return ::get_example_config();
    }
    bool validate_config(const json& value) const override
    {
        return value.contains("size_1d") &&
               value["size_1d"].is_number_integer() &&
               value.contains("num_batches") &&
               value["num_batches"].is_number_integer();
    }
    std::string describe_config(const json& test_case) const override
    {
        std::stringstream ss;
        ss << "tensor(" << test_case["size_1d"].get<gko::int64>() << ") x "
           << test_case["num_batches"].get<gko::int64>();
        return ss.str();
    }
    TensorState setup(std::shared_ptr<gko::Executor> exec,
                      json& test_case) const override
    {
        auto size_1d = test_case["size_1d"].get<gko::int64>();
        auto num_batches = test_case["num_batches"].get<gko::int64>();
        auto vec_size = gko::batch_dim<2>(
            num_batches, gko::dim<2>(size_1d * size_1d * size_1d, 1));
        auto engine = std::default_random_engine{42};
        TensorState state{
            gko::test::generate_random_batch_dense_matrix<
                gko::batch::matrix::Dense<vtype>>(
                num_batches, size_1d, size_1d,
                std::uniform_real_distribution<>(), engine, exec),
            gko::test::generate_random_batch_dense_matrix<
                gko::batch::MultiVector<vtype>>(
                num_batches, vec_size.get_common_size()[0], 1,
                std::uniform_real_distribution<>(), engine, exec),
            gko::batch::MultiVector<vtype>::create(exec, vec_size)};
        state.b->fill(gko::zero<vtype>());

        std::clog << "Matrix is of size (" << state.x->get_common_size()[0]
                  << ", " << state.x->get_common_size()[0] << ")" << std::endl;
        test_case["rows"] = state.x->get_common_size()[0];
        test_case["cols"] = state.x->get_common_size()[0];

        return state;
    }
    void run(std::shared_ptr<gko::Executor> exec, std::shared_ptr<Timer> timer,
             annotate_functor annotate, TensorState& state,
             const std::string& operation, json& operation_case) const override
    {
        auto run_impl = [&](const auto& op) {
            IterationControl ic{timer};
            // warm run
            {
                auto range = annotate("warmup", FLAGS_warmup > 0);
                for (auto _ : ic.warmup_run()) {
                    auto x_clone = clone(state.x);
                    exec->synchronize();
                    op->apply(state.b, x_clone);
                    exec->synchronize();
                }
            }

            // timed run
            auto x_clone = clone(state.x);
            for (auto _ : ic.run()) {
                auto range = annotate("repetition");
                op->apply(state.b, x_clone);
            }
            operation_case["time"] = ic.compute_time(FLAGS_timer_method);
            operation_case["repetitions"] = ic.get_num_repetitions();
        };

        using Dense = gko::batch::matrix::Dense<vtype>;
        using Csr = gko::batch::matrix::Csr<vtype>;

        auto tensor =
            std::make_shared<tensor::TensorLeft>(gko::clone(state.data_1d));
        if (operation == "matrix-free") {
            run_impl(tensor);
        } else if (operation == "matrix-dense") {
            run_impl(tensor::convert<Dense>(tensor));
        } else if (operation == "matrix-sparse") {
            auto size_1d = state.data_1d->get_common_size()[0];
            auto nnz = size_1d * size_1d * size_1d * size_1d;
            run_impl(tensor::convert<Csr>(tensor, nnz));
        } else {
            throw std::runtime_error("Unsupported operation: " + operation);
        }
    }
};


int main(int argc, char* argv[])
{
    std::string header =
        "A benchmark for measuring performance of Ginkgo's spmv.\n";
    std::string format = get_example_config();
    initialize_argument_parsing_matrix(&argc, &argv, header, format);

    auto exec = executor_factory.at(FLAGS_executor)(FLAGS_gpu_timer);

    std::string extra_information =
        "The apply formats are " + FLAGS_apply + ".";
    print_general_information(extra_information, exec);

    auto test_cases = json::parse(get_input_stream());

    auto benchmark = TensorBenchmark{};
    run_test_cases(benchmark, exec, get_timer(exec, FLAGS_gpu_timer),
                   test_cases);

    std::cout << std::setw(4) << test_cases << std::endl;
}
