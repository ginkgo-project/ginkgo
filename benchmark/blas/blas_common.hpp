// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/ginkgo.hpp>


#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <typeinfo>


#include "benchmark/utils/general.hpp"
#include "benchmark/utils/iteration_control.hpp"
#include "benchmark/utils/loggers.hpp"
#include "benchmark/utils/runner.hpp"
#include "benchmark/utils/timer.hpp"
#include "benchmark/utils/types.hpp"
#include "core/components/prefix_sum_kernels.hpp"


// Command-line arguments
DEFINE_string(
    operations, "copy,axpy,scal",
    "A comma-separated list of operations to benchmark.\nCandidates are"
    "BLAS algorithms:\n"
    "   copy (y = x),\n"
    "   axpy (y = y + a * x),\n"
    "   multiaxpy (like axpy, but a has one entry per column),\n"
    "   scal (y = a * y),\n"
    "   multiscal (like scal, but a has one entry per column),\n"
    "   dot (a = x' * y),"
    "   norm (a = sqrt(x' * x)),\n"
    "   mm (C = A * B),\n"
    "   gemm (C = a * A * B + b * C)\n"
    "Non-numerical algorithms:\n"
    "   prefix_sum32 (x_i <- sum_{j=0}^{i-1} x_i, 32 bit indices)\n"
    "   prefix_sum64 (                            64 bit indices)\n"
    "where A has dimensions n x k, B has dimensions k x m,\n"
    "C has dimensions n x m and x and y have dimensions n x r");


class BenchmarkOperation {
public:
    virtual ~BenchmarkOperation() = default;

    virtual gko::size_type get_flops() const = 0;
    virtual gko::size_type get_memory() const = 0;
    virtual void prepare(){};
    virtual void run() = 0;
};

template <typename Generator, typename T>
auto as_vector(const std::unique_ptr<T>& p)
{
    return gko::as<typename Generator::Vec>(p.get());
}


template <typename Generator>
class CopyOperation : public BenchmarkOperation {
public:
    CopyOperation(std::shared_ptr<const gko::Executor> exec,
                  const Generator& generator, gko::size_type rows,
                  gko::size_type cols, gko::size_type stride_in,
                  gko::size_type stride_out)
    {
        in_ = generator.create_multi_vector_strided(
            exec, gko::dim<2>{rows, cols}, stride_in);
        out_ = generator.create_multi_vector_strided(
            exec, gko::dim<2>{rows, cols}, stride_out);
        as_vector<Generator>(in_)->fill(1);
    }

    gko::size_type get_flops() const override
    {
        return in_->get_size()[0] * in_->get_size()[1];
    }

    gko::size_type get_memory() const override
    {
        return in_->get_size()[0] * in_->get_size()[1] * sizeof(etype) * 2;
    }

    void run() override
    {
        as_vector<Generator>(in_)->convert_to(as_vector<Generator>(out_));
    }

private:
    std::unique_ptr<gko::LinOp> in_;
    std::unique_ptr<gko::LinOp> out_;
};


template <typename Generator>
class AxpyOperation : public BenchmarkOperation {
public:
    AxpyOperation(std::shared_ptr<const gko::Executor> exec,
                  const Generator& generator, gko::size_type rows,
                  gko::size_type cols, gko::size_type stride_in,
                  gko::size_type stride_out, bool multi)
    {
        alpha_ = gko::matrix::Dense<etype>::create(
            exec, gko::dim<2>{1, multi ? cols : 1});
        x_ = generator.create_multi_vector_strided(
            exec, gko::dim<2>{rows, cols}, stride_in);
        y_ = generator.create_multi_vector_strided(
            exec, gko::dim<2>{rows, cols}, stride_out);
        alpha_->fill(1);
        as_vector<Generator>(x_)->fill(1);
    }

    gko::size_type get_flops() const override
    {
        return y_->get_size()[0] * y_->get_size()[1] * 2;
    }

    gko::size_type get_memory() const override
    {
        return y_->get_size()[0] * y_->get_size()[1] * sizeof(etype) * 3;
    }

    void prepare() override { as_vector<Generator>(y_)->fill(1); }

    void run() override { as_vector<Generator>(y_)->add_scaled(alpha_, x_); }

private:
    std::unique_ptr<gko::matrix::Dense<etype>> alpha_;
    std::unique_ptr<gko::LinOp> x_;
    std::unique_ptr<gko::LinOp> y_;
};


template <typename Generator>
class ScalOperation : public BenchmarkOperation {
public:
    ScalOperation(std::shared_ptr<const gko::Executor> exec,
                  const Generator& generator, gko::size_type rows,
                  gko::size_type cols, gko::size_type stride, bool multi)
    {
        alpha_ = gko::matrix::Dense<etype>::create(
            exec, gko::dim<2>{1, multi ? cols : 1});
        y_ = generator.create_multi_vector_strided(
            exec, gko::dim<2>{rows, cols}, stride);
        alpha_->fill(1);
    }

    gko::size_type get_flops() const override
    {
        return y_->get_size()[0] * y_->get_size()[1];
    }

    gko::size_type get_memory() const override
    {
        return y_->get_size()[0] * y_->get_size()[1] * sizeof(etype) * 2;
    }

    void prepare() override { as_vector<Generator>(y_)->fill(1); }

    void run() override { as_vector<Generator>(y_)->scale(alpha_); }

private:
    std::unique_ptr<gko::matrix::Dense<etype>> alpha_;
    std::unique_ptr<gko::LinOp> y_;
};


template <typename Generator>
class DotOperation : public BenchmarkOperation {
public:
    DotOperation(std::shared_ptr<const gko::Executor> exec,
                 const Generator& generator, gko::size_type rows,
                 gko::size_type cols, gko::size_type stride_x,
                 gko::size_type stride_y)
    {
        alpha_ = gko::matrix::Dense<etype>::create(exec, gko::dim<2>{1, cols});
        x_ = generator.create_multi_vector_strided(
            exec, gko::dim<2>{rows, cols}, stride_x);
        y_ = generator.create_multi_vector_strided(
            exec, gko::dim<2>{rows, cols}, stride_y);
        as_vector<Generator>(x_)->fill(1);
        as_vector<Generator>(y_)->fill(1);
    }

    gko::size_type get_flops() const override
    {
        return y_->get_size()[0] * y_->get_size()[1] * 2;
    }

    gko::size_type get_memory() const override
    {
        return y_->get_size()[0] * y_->get_size()[1] * sizeof(etype) * 2;
    }

    void run() override { as_vector<Generator>(x_)->compute_dot(y_, alpha_); }

private:
    std::unique_ptr<gko::matrix::Dense<etype>> alpha_;
    std::unique_ptr<gko::LinOp> x_;
    std::unique_ptr<gko::LinOp> y_;
};


template <typename Generator>
class NormOperation : public BenchmarkOperation {
public:
    NormOperation(std::shared_ptr<const gko::Executor> exec,
                  const Generator& generator, gko::size_type rows,
                  gko::size_type cols, gko::size_type stride)
    {
        alpha_ = gko::matrix::Dense<etype>::create(exec, gko::dim<2>{1, cols});
        y_ = generator.create_multi_vector_strided(
            exec, gko::dim<2>{rows, cols}, stride);
        as_vector<Generator>(y_)->fill(1);
    }

    gko::size_type get_flops() const override
    {
        return y_->get_size()[0] * y_->get_size()[1] * 2;
    }

    gko::size_type get_memory() const override
    {
        return y_->get_size()[0] * y_->get_size()[1] * sizeof(etype);
    }

    void run() override { as_vector<Generator>(y_)->compute_norm2(alpha_); }

private:
    std::unique_ptr<gko::matrix::Dense<etype>> alpha_;
    std::unique_ptr<gko::LinOp> y_;
};


template <typename Generator>
class ApplyOperation : public BenchmarkOperation {
public:
    ApplyOperation(std::shared_ptr<const gko::Executor> exec,
                   const Generator& generator, gko::size_type n,
                   gko::size_type k, gko::size_type m, gko::size_type stride_A,
                   gko::size_type stride_B, gko::size_type stride_C)
    {
        A_ = generator.create_multi_vector_strided(exec, gko::dim<2>{n, k},
                                                   stride_A);
        B_ = generator.create_multi_vector_strided(exec, gko::dim<2>{k, m},
                                                   stride_B);
        C_ = generator.create_multi_vector_strided(exec, gko::dim<2>{n, m},
                                                   stride_C);
        as_vector<Generator>(A_)->fill(1);
        as_vector<Generator>(B_)->fill(1);
    }

    gko::size_type get_flops() const override
    {
        return A_->get_size()[0] * A_->get_size()[1] * B_->get_size()[1] * 2;
    }

    gko::size_type get_memory() const override
    {
        return (A_->get_size()[0] * A_->get_size()[1] +
                B_->get_size()[0] * B_->get_size()[1] +
                C_->get_size()[0] * C_->get_size()[1]) *
               sizeof(etype);
    }

    void run() override { A_->apply(B_, C_); }

private:
    std::unique_ptr<gko::LinOp> A_;
    std::unique_ptr<gko::LinOp> B_;
    std::unique_ptr<gko::LinOp> C_;
};


template <typename Generator>
class AdvancedApplyOperation : public BenchmarkOperation {
public:
    AdvancedApplyOperation(std::shared_ptr<const gko::Executor> exec,
                           const Generator& generator, gko::size_type n,
                           gko::size_type k, gko::size_type m,
                           gko::size_type stride_A, gko::size_type stride_B,
                           gko::size_type stride_C)
    {
        A_ = generator.create_multi_vector_strided(exec, gko::dim<2>{n, k},
                                                   stride_A);
        B_ = generator.create_multi_vector_strided(exec, gko::dim<2>{k, m},
                                                   stride_B);
        C_ = generator.create_multi_vector_strided(exec, gko::dim<2>{n, m},
                                                   stride_C);
        alpha_ = gko::matrix::Dense<etype>::create(exec, gko::dim<2>{1, 1});
        beta_ = gko::matrix::Dense<etype>::create(exec, gko::dim<2>{1, 1});
        as_vector<Generator>(A_)->fill(1);
        as_vector<Generator>(B_)->fill(1);
        alpha_->fill(1);
        beta_->fill(1);
    }

    gko::size_type get_flops() const override
    {
        return A_->get_size()[0] * A_->get_size()[1] * B_->get_size()[1] * 2 +
               C_->get_size()[0] * C_->get_size()[1] * 3;
    }

    gko::size_type get_memory() const override
    {
        return (A_->get_size()[0] * A_->get_size()[1] +
                B_->get_size()[0] * B_->get_size()[1] +
                C_->get_size()[0] * C_->get_size()[1] * 2) *
               sizeof(etype);
    }

    void run() override { A_->apply(alpha_, B_, beta_, C_); }

private:
    std::unique_ptr<gko::matrix::Dense<etype>> alpha_;
    std::unique_ptr<gko::matrix::Dense<etype>> beta_;
    std::unique_ptr<gko::LinOp> A_;
    std::unique_ptr<gko::LinOp> B_;
    std::unique_ptr<gko::LinOp> C_;
};


GKO_REGISTER_OPERATION(prefix_sum_nonnegative,
                       components::prefix_sum_nonnegative);


template <typename IndexType>
class PrefixSumOperation : public BenchmarkOperation {
public:
    PrefixSumOperation(std::shared_ptr<const gko::Executor> exec,
                       gko::size_type n)
        : array_{exec, n}
    {
        array_.fill(0);
    }

    gko::size_type get_flops() const override { return 0; }

    gko::size_type get_memory() const override
    {
        return 2 * sizeof(IndexType) * array_.get_size();
    }

    void run() override
    {
        array_.get_executor()->run(
            make_prefix_sum_nonnegative(array_.get_data(), array_.get_size()));
    }

private:
    gko::array<IndexType> array_;
};


struct dimensions {
    gko::size_type n;
    gko::size_type k;
    gko::size_type m;
    gko::size_type r;
    gko::size_type stride_x;
    gko::size_type stride_y;
    gko::size_type stride_A;
    gko::size_type stride_B;
    gko::size_type stride_C;
};


struct BlasBenchmark : Benchmark<dimensions> {
    using map_type =
        std::map<std::string,
                 std::function<std::unique_ptr<BenchmarkOperation>(
                     std::shared_ptr<const gko::Executor>, dimensions)>>;
    map_type operation_map;
    std::vector<std::string> operations;
    std::string name;
    bool do_print;

    BlasBenchmark(map_type operation_map, bool do_print = true)
        : operation_map{std::move(operation_map)},
          name{"blas"},
          operations{split(FLAGS_operations)},
          do_print{do_print}
    {}

    const std::string& get_name() const override { return name; }

    const std::vector<std::string>& get_operations() const override
    {
        return operations;
    }

    bool should_print() const override { return do_print; }

    std::string get_example_config() const override
    {
        return json::parse(R"([{"n": 100}, {"n": 200, "m": 200, "k": 200}])")
            .dump(4);
    }

    bool validate_config(const json& value) const override
    {
        return value.contains("n") && value["n"].is_number_integer();
    }

    std::string describe_config(const json& test_case) const override
    {
        std::stringstream ss;
        auto optional_output = [&](const char* name) {
            if (test_case.contains(name) &&
                test_case[name].is_number_integer()) {
                ss << name << " = " << test_case[name].get<gko::int64>() << " ";
            }
        };
        optional_output("n");
        optional_output("k");
        optional_output("m");
        optional_output("r");
        optional_output("stride");
        optional_output("stride_x");
        optional_output("stride_y");
        optional_output("stride_A");
        optional_output("stride_B");
        optional_output("stride_C");
        return ss.str();
    }

    dimensions setup(std::shared_ptr<gko::Executor> exec,
                     json& test_case) const override
    {
        auto get_optional = [](json& obj, const char* name,
                               gko::size_type default_value) -> gko::size_type {
            if (obj.contains(name)) {
                return obj[name].get<gko::uint64>();
            } else {
                return default_value;
            }
        };

        dimensions result;
        result.n = test_case["n"].get<gko::int64>();
        result.k = get_optional(test_case, "k", result.n);
        result.m = get_optional(test_case, "m", result.n);
        result.r = get_optional(test_case, "r", 1);
        if (test_case.contains("stride")) {
            result.stride_x = test_case["stride"].get<gko::int64>();
            result.stride_y = result.stride_x;
        } else {
            result.stride_x = get_optional(test_case, "stride_x", result.r);
            result.stride_y = get_optional(test_case, "stride_y", result.r);
        }
        result.stride_A = get_optional(test_case, "stride_A", result.k);
        result.stride_B = get_optional(test_case, "stride_B", result.m);
        result.stride_C = get_optional(test_case, "stride_C", result.m);
        return result;
    }


    void run(std::shared_ptr<gko::Executor> exec, std::shared_ptr<Timer> timer,
             annotate_functor annotate, dimensions& dims,
             const std::string& operation_name,
             json& operation_case) const override
    {
        auto op = operation_map.at(operation_name)(exec, dims);

        IterationControl ic(timer);

        // warm run
        {
            auto range = annotate("warmup", FLAGS_warmup > 0);
            for (auto _ : ic.warmup_run()) {
                op->prepare();
                exec->synchronize();
                op->run();
                exec->synchronize();
            }
        }

        // timed run
        op->prepare();
        for (auto _ : ic.run()) {
            auto range = annotate("repetition");
            op->run();
        }
        const auto runtime = ic.compute_time(FLAGS_timer_method);
        const auto flops = static_cast<double>(op->get_flops());
        const auto mem = static_cast<double>(op->get_memory());
        const auto repetitions = ic.get_num_repetitions();
        operation_case["time"] = runtime;
        operation_case["flops"] = flops / runtime;
        operation_case["bandwidth"] = mem / runtime;
        operation_case["repetitions"] = repetitions;
    }
};
