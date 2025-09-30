// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <typeinfo>

#include <ginkgo/ginkgo.hpp>

#include "benchmark/utils/general.hpp"
#include "benchmark/utils/iteration_control.hpp"
#include "benchmark/utils/loggers.hpp"
#include "benchmark/utils/runner.hpp"
#include "benchmark/utils/timer.hpp"
#include "benchmark/utils/types.hpp"
#include "core/components/prefix_sum_kernels.hpp"


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
        auto size = gko::dim<2>{rows, cols};
        in_ = generator.create_multi_vector_strided(
            exec, size, generator.create_default_local_size(size), stride_in);
        out_ = generator.create_multi_vector_strided(
            exec, size, generator.create_default_local_size(size), stride_out);
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
        auto size = gko::dim<2>{rows, cols};
        alpha_ = gko::matrix::Dense<etype>::create(
            exec, gko::dim<2>{1, multi ? cols : 1});
        x_ = generator.create_multi_vector_strided(
            exec, size, generator.create_default_local_size(size), stride_in);
        y_ = generator.create_multi_vector_strided(
            exec, size, generator.create_default_local_size(size), stride_out);
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
class SubScaledOperation : public BenchmarkOperation {
public:
    SubScaledOperation(std::shared_ptr<const gko::Executor> exec,
                       const Generator& generator, gko::size_type rows,
                       gko::size_type cols, gko::size_type stride_in,
                       gko::size_type stride_out, bool multi)
    {
        auto size = gko::dim<2>{rows, cols};
        alpha_ = gko::matrix::Dense<etype>::create(
            exec, gko::dim<2>{1, multi ? cols : 1});
        x_ = generator.create_multi_vector_strided(
            exec, size, generator.create_default_local_size(size), stride_in);
        y_ = generator.create_multi_vector_strided(
            exec, size, generator.create_default_local_size(size), stride_out);
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

    void run() override { as_vector<Generator>(y_)->sub_scaled(alpha_, x_); }

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
        auto size = gko::dim<2>{rows, cols};
        alpha_ = gko::matrix::Dense<etype>::create(
            exec, gko::dim<2>{1, multi ? cols : 1});
        y_ = generator.create_multi_vector_strided(
            exec, size, generator.create_default_local_size(size), stride);
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
        auto size = gko::dim<2>{rows, cols};
        alpha_ = gko::matrix::Dense<etype>::create(exec, gko::dim<2>{1, cols});
        x_ = generator.create_multi_vector_strided(
            exec, size, generator.create_default_local_size(size), stride_x);
        y_ = generator.create_multi_vector_strided(
            exec, size, generator.create_default_local_size(size), stride_y);
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
        auto size = gko::dim<2>{rows, cols};
        alpha_ = gko::matrix::Dense<etype>::create(exec, gko::dim<2>{1, cols});
        y_ = generator.create_multi_vector_strided(
            exec, size, generator.create_default_local_size(size), stride);
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
        // Since dense distributed matrices are not supported we can use
        // local_size == global_size
        A_ = generator.create_multi_vector_strided(exec, gko::dim<2>{n, k},
                                                   gko::dim<2>{n, k}, stride_A);
        B_ = generator.create_multi_vector_strided(exec, gko::dim<2>{k, m},
                                                   gko::dim<2>{k, m}, stride_B);
        C_ = generator.create_multi_vector_strided(exec, gko::dim<2>{n, m},
                                                   gko::dim<2>{n, m}, stride_C);
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
        // Since dense distributed matrices are not supported we can use
        // local_size == global_size
        A_ = generator.create_multi_vector_strided(exec, gko::dim<2>{n, k},
                                                   gko::dim<2>{n, k}, stride_A);
        B_ = generator.create_multi_vector_strided(exec, gko::dim<2>{k, m},
                                                   gko::dim<2>{k, m}, stride_B);
        C_ = generator.create_multi_vector_strided(exec, gko::dim<2>{n, m},
                                                   gko::dim<2>{n, m}, stride_C);
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
    std::string name;
    bool do_print;

    BlasBenchmark(map_type operation_map, bool do_print = true)
        : operation_map{std::move(operation_map)},
          name{"blas"},
          do_print{do_print}
    {}

    const std::string& get_name() const override { return name; }


    bool should_print() const override { return do_print; }

    dimensions setup(std::shared_ptr<gko::Executor> exec,
                     json& test_case) const override
    {
        auto get_optional = [](json& obj, const char* name,
                               gko::size_type default_value) -> gko::size_type {
            if (!obj.contains(name)) {
                obj[name] = default_value;
            }
            return obj[name].get<gko::uint64>();
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
             const json& operation_case, json& result_case) const override
    {
        auto op = operation_map.at(
            operation_case["operation"].get<std::string>())(exec, dims);

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
        result_case["time"] = runtime;
        result_case["flops"] = flops / runtime;
        result_case["bandwidth"] = mem / runtime;
        result_case["repetitions"] = repetitions;
    }

    void postprocess(json& test_cases) const override
    {
        std::map<json, json> same_operators;
        for (const auto& test_case : test_cases) {
            auto case_operator = test_case;
            case_operator.erase("operation");
            case_operator.erase(name);
            same_operators.try_emplace(case_operator, json::object());
            same_operators[case_operator][test_case["operation"]] =
                test_case[name];
        }
        auto merged_cases = json::array();
        for (const auto& [test_case, results] : same_operators) {
            merged_cases.push_back(test_case);
            merged_cases.back()[name] = results;
        }
        test_cases = std::move(merged_cases);
    }
};
