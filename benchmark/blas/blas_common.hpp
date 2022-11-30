/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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
#include <iomanip>
#include <iostream>
#include <typeinfo>


#include "benchmark/utils/general.hpp"
#include "benchmark/utils/loggers.hpp"
#include "benchmark/utils/timer.hpp"
#include "benchmark/utils/types.hpp"


// Command-line arguments
DEFINE_string(
    operations, "copy,axpy,scal",
    "A comma-separated list of BLAS operations to benchmark.\nCandidates are"
    "   copy (y = x),\n"
    "   axpy (y = y + a * x),\n"
    "   multiaxpy (like axpy, but a has one entry per column),\n"
    "   scal (y = a * y),\n"
    "   multiscal (like scal, but a has one entry per column),\n"
    "   dot (a = x' * y),"
    "   norm (a = sqrt(x' * x)),\n"
    "   mm (C = A * B),\n"
    "   gemm (C = a * A * B + b * C)\n"
    "where A has dimensions n x k, B has dimensions k x m,\n"
    "C has dimensions n x m and x and y have dimensions n x r");


std::string example_config = R"(
  [
    { "n": 100 },
    { "n": 200, "m": 200, "k": 200 }
  ]
)";


class BenchmarkOperation {
public:
    virtual ~BenchmarkOperation() = default;

    virtual gko::size_type get_flops() const = 0;
    virtual gko::size_type get_memory() const = 0;
    virtual void prepare(){};
    virtual void run() = 0;
};

template <typename Generator, typename T>
auto as_vector(T& p)
{
    return gko::as<typename Generator::Vec>(gko::lend(p));
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

    void run() override
    {
        as_vector<Generator>(y_)->add_scaled(lend(alpha_), lend(x_));
    }

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

    void run() override { as_vector<Generator>(y_)->scale(lend(alpha_)); }

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

    void run() override
    {
        as_vector<Generator>(x_)->compute_dot(lend(y_), lend(alpha_));
    }

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

    void run() override
    {
        as_vector<Generator>(y_)->compute_norm2(lend(alpha_));
    }

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

    void run() override { A_->apply(lend(B_), lend(C_)); }

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

    void run() override
    {
        A_->apply(lend(alpha_), lend(B_), lend(beta_), lend(C_));
    }

private:
    std::unique_ptr<gko::matrix::Dense<etype>> alpha_;
    std::unique_ptr<gko::matrix::Dense<etype>> beta_;
    std::unique_ptr<gko::LinOp> A_;
    std::unique_ptr<gko::LinOp> B_;
    std::unique_ptr<gko::LinOp> C_;
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


dimensions parse_dims(rapidjson::Value& test_case)
{
    auto get_optional = [](rapidjson::Value& obj, const char* name,
                           gko::size_type default_value) -> gko::size_type {
        if (obj.HasMember(name)) {
            return obj[name].GetUint64();
        } else {
            return default_value;
        }
    };

    dimensions result;
    result.n = test_case["n"].GetInt64();
    result.k = get_optional(test_case, "k", result.n);
    result.m = get_optional(test_case, "m", result.n);
    result.r = get_optional(test_case, "r", 1);
    if (test_case.HasMember("stride")) {
        result.stride_x = test_case["stride"].GetInt64();
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


template <typename OpMap>
void apply_blas(const char* operation_name, std::shared_ptr<gko::Executor> exec,
                std::shared_ptr<Timer> timer, const OpMap& operation_map,
                rapidjson::Value& test_case,
                rapidjson::MemoryPoolAllocator<>& allocator)
{
    try {
        auto& blas_case = test_case["blas"];
        add_or_set_member(blas_case, operation_name,
                          rapidjson::Value(rapidjson::kObjectType), allocator);

        auto op = operation_map.at(operation_name)(exec, parse_dims(test_case));

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
        add_or_set_member(blas_case[operation_name], "time", runtime,
                          allocator);
        add_or_set_member(blas_case[operation_name], "flops", flops / runtime,
                          allocator);
        add_or_set_member(blas_case[operation_name], "bandwidth", mem / runtime,
                          allocator);
        add_or_set_member(blas_case[operation_name], "repetitions", repetitions,
                          allocator);

        // compute and write benchmark data
        add_or_set_member(blas_case[operation_name], "completed", true,
                          allocator);
    } catch (const std::exception& e) {
        add_or_set_member(test_case["blas"][operation_name], "completed", false,
                          allocator);
        if (FLAGS_keep_errors) {
            rapidjson::Value msg_value;
            msg_value.SetString(e.what(), allocator);
            add_or_set_member(test_case["blas"][operation_name], "error",
                              msg_value, allocator);
        }
        std::cerr << "Error when processing test case " << test_case << "\n"
                  << "what(): " << e.what() << std::endl;
    }
}


template <typename OpMap>
void run_blas_benchmarks(std::shared_ptr<gko::Executor> exec,
                         std::shared_ptr<Timer> timer,
                         const OpMap& operation_map,
                         rapidjson::Document& test_cases, bool do_print)
{
    auto operations = split(FLAGS_operations, ',');
    auto& allocator = test_cases.GetAllocator();

    for (auto& test_case : test_cases.GetArray()) {
        try {
            // set up benchmark
            if (!test_case.HasMember("blas")) {
                test_case.AddMember("blas",
                                    rapidjson::Value(rapidjson::kObjectType),
                                    allocator);
            }
            auto& blas_case = test_case["blas"];
            if (!FLAGS_overwrite &&
                all_of(begin(operations), end(operations),
                       [&blas_case](const std::string& s) {
                           return blas_case.HasMember(s.c_str());
                       })) {
                continue;
            }
            if (do_print) {
                std::clog << "Running test case: " << test_case << std::endl;
            }

            for (const auto& operation_name : operations) {
                apply_blas(operation_name.c_str(), exec, timer, operation_map,
                           test_case, allocator);

                if (do_print) {
                    std::clog << "Current state:" << std::endl
                              << test_cases << std::endl;

                    backup_results(test_cases);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error setting up benchmark, what(): " << e.what()
                      << std::endl;
        }
    }
}
