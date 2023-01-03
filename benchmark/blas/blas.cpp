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


class BenchmarkOperation {
public:
    virtual ~BenchmarkOperation() = default;

    virtual gko::size_type get_flops() const = 0;
    virtual gko::size_type get_memory() const = 0;
    virtual void prepare(){};
    virtual void run() = 0;
};


class CopyOperation : public BenchmarkOperation {
public:
    CopyOperation(std::shared_ptr<const gko::Executor> exec,
                  gko::size_type rows, gko::size_type cols,
                  gko::size_type istride, gko::size_type ostride)
    {
        in_ = gko::matrix::Dense<etype>::create(exec, gko::dim<2>{rows, cols},
                                                istride);
        out_ = gko::matrix::Dense<etype>::create(exec, gko::dim<2>{rows, cols},
                                                 ostride);
        in_->fill(1);
    }

    gko::size_type get_flops() const override
    {
        return in_->get_size()[0] * in_->get_size()[1];
    }

    gko::size_type get_memory() const override
    {
        return in_->get_size()[0] * in_->get_size()[1] * sizeof(etype) * 2;
    }

    void run() override { in_->convert_to(lend(out_)); }

private:
    std::unique_ptr<gko::matrix::Dense<etype>> in_;
    std::unique_ptr<gko::matrix::Dense<etype>> out_;
};


class AxpyOperation : public BenchmarkOperation {
public:
    AxpyOperation(std::shared_ptr<const gko::Executor> exec,
                  gko::size_type rows, gko::size_type cols,
                  gko::size_type stride_in, gko::size_type stride_out,
                  bool multi)
    {
        alpha_ = gko::matrix::Dense<etype>::create(
            exec, gko::dim<2>{1, multi ? cols : 1});
        x_ = gko::matrix::Dense<etype>::create(exec, gko::dim<2>{rows, cols},
                                               stride_in);
        y_ = gko::matrix::Dense<etype>::create(exec, gko::dim<2>{rows, cols},
                                               stride_out);
        alpha_->fill(1);
        x_->fill(1);
    }

    gko::size_type get_flops() const override
    {
        return y_->get_size()[0] * y_->get_size()[1] * 2;
    }

    gko::size_type get_memory() const override
    {
        return y_->get_size()[0] * y_->get_size()[1] * sizeof(etype) * 3;
    }

    void prepare() override { y_->fill(1); }

    void run() override { y_->add_scaled(lend(alpha_), lend(x_)); }

private:
    std::unique_ptr<gko::matrix::Dense<etype>> alpha_;
    std::unique_ptr<gko::matrix::Dense<etype>> x_;
    std::unique_ptr<gko::matrix::Dense<etype>> y_;
};


class ScalOperation : public BenchmarkOperation {
public:
    ScalOperation(std::shared_ptr<const gko::Executor> exec,
                  gko::size_type rows, gko::size_type cols,
                  gko::size_type stride, bool multi)
    {
        alpha_ = gko::matrix::Dense<etype>::create(
            exec, gko::dim<2>{1, multi ? cols : 1});
        y_ = gko::matrix::Dense<etype>::create(exec, gko::dim<2>{rows, cols},
                                               stride);
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

    void prepare() override { y_->fill(1); }

    void run() override { y_->scale(lend(alpha_)); }

private:
    std::unique_ptr<gko::matrix::Dense<etype>> alpha_;
    std::unique_ptr<gko::matrix::Dense<etype>> y_;
};


class DotOperation : public BenchmarkOperation {
public:
    DotOperation(std::shared_ptr<const gko::Executor> exec, gko::size_type rows,
                 gko::size_type cols, gko::size_type stride_x,
                 gko::size_type stride_y)
    {
        alpha_ = gko::matrix::Dense<etype>::create(exec, gko::dim<2>{1, cols});
        x_ = gko::matrix::Dense<etype>::create(exec, gko::dim<2>{rows, cols},
                                               stride_x);
        y_ = gko::matrix::Dense<etype>::create(exec, gko::dim<2>{rows, cols},
                                               stride_y);
        x_->fill(1);
        y_->fill(1);
    }

    gko::size_type get_flops() const override
    {
        return y_->get_size()[0] * y_->get_size()[1] * 2;
    }

    gko::size_type get_memory() const override
    {
        return y_->get_size()[0] * y_->get_size()[1] * sizeof(etype) * 2;
    }

    void run() override { x_->compute_dot(lend(y_), lend(alpha_)); }

private:
    std::unique_ptr<gko::matrix::Dense<etype>> alpha_;
    std::unique_ptr<gko::matrix::Dense<etype>> x_;
    std::unique_ptr<gko::matrix::Dense<etype>> y_;
};


class NormOperation : public BenchmarkOperation {
public:
    NormOperation(std::shared_ptr<const gko::Executor> exec,
                  gko::size_type rows, gko::size_type cols,
                  gko::size_type stride)
    {
        alpha_ = gko::matrix::Dense<etype>::create(exec, gko::dim<2>{1, cols});
        y_ = gko::matrix::Dense<etype>::create(exec, gko::dim<2>{rows, cols},
                                               stride);
        y_->fill(1);
    }

    gko::size_type get_flops() const override
    {
        return y_->get_size()[0] * y_->get_size()[1] * 2;
    }

    gko::size_type get_memory() const override
    {
        return y_->get_size()[0] * y_->get_size()[1] * sizeof(etype);
    }

    void run() override { y_->compute_norm2(lend(alpha_)); }

private:
    std::unique_ptr<gko::matrix::Dense<etype>> alpha_;
    std::unique_ptr<gko::matrix::Dense<etype>> y_;
};


class ApplyOperation : public BenchmarkOperation {
public:
    ApplyOperation(std::shared_ptr<const gko::Executor> exec, gko::size_type n,
                   gko::size_type k, gko::size_type m, gko::size_type stride_A,
                   gko::size_type stride_B, gko::size_type stride_C)
    {
        A_ = gko::matrix::Dense<etype>::create(exec, gko::dim<2>{n, k},
                                               stride_A);
        B_ = gko::matrix::Dense<etype>::create(exec, gko::dim<2>{k, m},
                                               stride_B);
        C_ = gko::matrix::Dense<etype>::create(exec, gko::dim<2>{n, m},
                                               stride_C);
        A_->fill(1);
        B_->fill(1);
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
    std::unique_ptr<gko::matrix::Dense<etype>> A_;
    std::unique_ptr<gko::matrix::Dense<etype>> B_;
    std::unique_ptr<gko::matrix::Dense<etype>> C_;
};


class AdvancedApplyOperation : public BenchmarkOperation {
public:
    AdvancedApplyOperation(std::shared_ptr<const gko::Executor> exec,
                           gko::size_type n, gko::size_type k, gko::size_type m,
                           gko::size_type stride_A, gko::size_type stride_B,
                           gko::size_type stride_C)
    {
        A_ = gko::matrix::Dense<etype>::create(exec, gko::dim<2>{n, k},
                                               stride_A);
        B_ = gko::matrix::Dense<etype>::create(exec, gko::dim<2>{k, m},
                                               stride_B);
        C_ = gko::matrix::Dense<etype>::create(exec, gko::dim<2>{n, m},
                                               stride_C);
        alpha_ = gko::matrix::Dense<etype>::create(exec, gko::dim<2>{1, 1});
        beta_ = gko::matrix::Dense<etype>::create(exec, gko::dim<2>{1, 1});
        A_->fill(1);
        B_->fill(1);
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
    std::unique_ptr<gko::matrix::Dense<etype>> A_;
    std::unique_ptr<gko::matrix::Dense<etype>> B_;
    std::unique_ptr<gko::matrix::Dense<etype>> C_;
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


gko::size_type get_optional(rapidjson::Value& obj, const char* name,
                            gko::size_type default_value)
{
    if (obj.HasMember(name)) {
        return obj[name].GetUint64();
    } else {
        return default_value;
    }
}


dimensions parse_dims(rapidjson::Value& test_case)
{
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


std::map<std::string, std::function<std::unique_ptr<BenchmarkOperation>(
                          std::shared_ptr<const gko::Executor>, dimensions)>>
    operation_map{
        {"copy",
         [](std::shared_ptr<const gko::Executor> exec, dimensions dims) {
             return std::make_unique<CopyOperation>(
                 exec, dims.n, dims.r, dims.stride_x, dims.stride_y);
         }},
        {"axpy",
         [](std::shared_ptr<const gko::Executor> exec, dimensions dims) {
             return std::make_unique<AxpyOperation>(
                 exec, dims.n, dims.r, dims.stride_x, dims.stride_y, false);
         }},
        {"multiaxpy",
         [](std::shared_ptr<const gko::Executor> exec, dimensions dims) {
             return std::make_unique<AxpyOperation>(
                 exec, dims.n, dims.r, dims.stride_x, dims.stride_y, true);
         }},
        {"scal",
         [](std::shared_ptr<const gko::Executor> exec, dimensions dims) {
             return std::make_unique<ScalOperation>(exec, dims.n, dims.r,
                                                    dims.stride_y, false);
         }},
        {"multiscal",
         [](std::shared_ptr<const gko::Executor> exec, dimensions dims) {
             return std::make_unique<ScalOperation>(exec, dims.n, dims.r,
                                                    dims.stride_y, true);
         }},
        {"dot",
         [](std::shared_ptr<const gko::Executor> exec, dimensions dims) {
             return std::make_unique<DotOperation>(
                 exec, dims.n, dims.r, dims.stride_x, dims.stride_y);
         }},
        {"norm",
         [](std::shared_ptr<const gko::Executor> exec, dimensions dims) {
             return std::make_unique<NormOperation>(exec, dims.n, dims.r,
                                                    dims.stride_y);
         }},
        {"mm",
         [](std::shared_ptr<const gko::Executor> exec, dimensions dims) {
             return std::make_unique<ApplyOperation>(
                 exec, dims.n, dims.k, dims.m, dims.stride_A, dims.stride_B,
                 dims.stride_C);
         }},
        {"gemm",
         [](std::shared_ptr<const gko::Executor> exec, dimensions dims) {
             return std::make_unique<AdvancedApplyOperation>(
                 exec, dims.n, dims.k, dims.m, dims.stride_A, dims.stride_B,
                 dims.stride_C);
         }}};


void apply_blas(const char* operation_name, std::shared_ptr<gko::Executor> exec,
                rapidjson::Value& test_case,
                rapidjson::MemoryPoolAllocator<>& allocator)
{
    try {
        auto& blas_case = test_case["blas"];
        add_or_set_member(blas_case, operation_name,
                          rapidjson::Value(rapidjson::kObjectType), allocator);

        auto op = operation_map[operation_name](exec, parse_dims(test_case));

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


int main(int argc, char* argv[])
{
    std::string header =
        "A benchmark for measuring performance of Ginkgo's BLAS-like "
        "operations.\nParameters for a benchmark case are:\n"
        "    n: number of rows for vectors and gemm output (required)\n"
        "    r: number of columns for vectors (optional, default 1)\n"
        "    m: number of columns for gemm output (optional, default n)\n"
        "    k: inner dimension of the gemm (optional, default n)\n"
        "    stride: storage stride for both vectors (optional, default r)\n"
        "    stride_x: stride for input vector x (optional, default r)\n"
        "    stride_y: stride for in/out vector y (optional, default r)\n"
        "    stride_A: stride for A matrix in gemm (optional, default k)\n"
        "    stride_B: stride for B matrix in gemm (optional, default m)\n"
        "    stride_C: stride for C matrix in gemm (optional, default m)\n";
    std::string format = std::string() + "  [\n    { \"n\": 100 },\n" +
                         "    { \"n\": 200, \"m\": 200, \"k\": 200 }\n" +
                         "  ]\n\n";
    initialize_argument_parsing(&argc, &argv, header, format);

    std::string extra_information = "The operations are " + FLAGS_operations;
    print_general_information(extra_information);
    auto exec = executor_factory.at(FLAGS_executor)(FLAGS_gpu_timer);
    auto engine = get_engine();
    auto operations = split(FLAGS_operations, ',');

    rapidjson::IStreamWrapper jcin(std::cin);
    rapidjson::Document test_cases;
    test_cases.ParseStream(jcin);
    if (!test_cases.IsArray()) {
        std::cerr
            << "Input has to be a JSON array of benchmark configurations:\n"
            << format;
        std::exit(1);
    }

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
            std::clog << "Running test case: " << test_case << std::endl;

            for (const auto& operation_name : operations) {
                apply_blas(operation_name.c_str(), exec, test_case, allocator);
                std::clog << "Current state:" << std::endl
                          << test_cases << std::endl;
                backup_results(test_cases);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error setting up benchmark, what(): " << e.what()
                      << std::endl;
        }
    }

    std::cout << test_cases << std::endl;
}
