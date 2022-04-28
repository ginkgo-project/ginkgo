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
#include <iomanip>
#include <iostream>
#include <typeinfo>


#include "benchmark/utils/general.hpp"
#include "benchmark/utils/loggers.hpp"
#include "benchmark/utils/timer.hpp"
#include "benchmark/utils/types.hpp"


// Command-line arguments
DEFINE_string(operations, "dot,norm",
              "A comma-separated list of multivector operations to "
              "benchmark.\nCandidates are"
              "   copy (y = x),\n"
              "   axpy (y = y + a * x),\n"
              "   multiaxpy (like axpy, but a has one entry per column),\n"
              "   scal (y = a * y),\n"
              "   multiscal (like scal, but a has one entry per column),\n"
              "   dot (a = x' * y),\n"
              "   norm (a = sqrt(x' * x)),\n"
              "x and y have dimensions n x r.\n"
              "Note that only 'dot' and 'norm' require communication.\n");
DEFINE_string(input, "input.json", "Input json file to read from.");


auto local_rows(gko::size_type global_rows, int num_procs, int rank)
{
    auto uniform_local_size = global_rows / num_procs;
    auto use_rest = rank < (global_rows % num_procs);
    return uniform_local_size + (use_rest ? 1 : 0);
}


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
                  gko::mpi::communicator comm, gko::size_type rows,
                  gko::size_type cols, gko::size_type istride,
                  gko::size_type ostride)
    {
        in_ = gko::experimental::distributed::Vector<etype>::create(
            exec, comm, gko::dim<2>{rows, cols},
            gko::dim<2>{local_rows(rows, comm.size(), comm.rank()), cols},
            istride);
        out_ = gko::experimental::distributed::Vector<etype>::create(
            exec, comm, gko::dim<2>{rows, cols},
            gko::dim<2>{local_rows(rows, comm.size(), comm.rank()), cols},
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

    void run() override { in_->convert_to(gko::lend(out_)); }

private:
    std::unique_ptr<gko::experimental::distributed::Vector<etype>> in_;
    std::unique_ptr<gko::experimental::distributed::Vector<etype>> out_;
};


class AxpyOperation : public BenchmarkOperation {
public:
    AxpyOperation(std::shared_ptr<const gko::Executor> exec,
                  gko::mpi::communicator comm, gko::size_type rows,
                  gko::size_type cols, gko::size_type stride_in,
                  gko::size_type stride_out, bool multi)
    {
        alpha_ = gko::matrix::Dense<etype>::create(
            exec, gko::dim<2>{1, multi ? cols : 1});
        x_ = gko::experimental::distributed::Vector<etype>::create(
            exec, comm, gko::dim<2>{rows, cols},
            gko::dim<2>{local_rows(rows, comm.size(), comm.rank()), cols},
            stride_in);
        y_ = gko::experimental::distributed::Vector<etype>::create(
            exec, comm, gko::dim<2>{rows, cols},
            gko::dim<2>{local_rows(rows, comm.size(), comm.rank()), cols},
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

    void run() override { y_->add_scaled(gko::lend(alpha_), gko::lend(x_)); }

private:
    std::unique_ptr<gko::matrix::Dense<etype>> alpha_;
    std::unique_ptr<gko::experimental::distributed::Vector<etype>> x_;
    std::unique_ptr<gko::experimental::distributed::Vector<etype>> y_;
};


class ScalOperation : public BenchmarkOperation {
public:
    ScalOperation(std::shared_ptr<const gko::Executor> exec,
                  gko::mpi::communicator comm, gko::size_type rows,
                  gko::size_type cols, gko::size_type stride, bool multi)
    {
        alpha_ = gko::matrix::Dense<etype>::create(
            exec, gko::dim<2>{1, multi ? cols : 1});
        y_ = gko::experimental::distributed::Vector<etype>::create(
            exec, comm, gko::dim<2>{rows, cols},
            gko::dim<2>{local_rows(rows, comm.size(), comm.rank()), cols},
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

    void run() override { y_->scale(gko::lend(alpha_)); }

private:
    std::unique_ptr<gko::matrix::Dense<etype>> alpha_;
    std::unique_ptr<gko::experimental::distributed::Vector<etype>> y_;
};


class DotOperation : public BenchmarkOperation {
public:
    DotOperation(std::shared_ptr<const gko::Executor> exec,
                 gko::mpi::communicator comm, gko::size_type rows,
                 gko::size_type cols, gko::size_type stride_x,
                 gko::size_type stride_y)
    {
        alpha_ = gko::matrix::Dense<etype>::create(exec, gko::dim<2>{1, cols});
        x_ = gko::experimental::distributed::Vector<etype>::create(
            exec, comm, gko::dim<2>{rows, cols},
            gko::dim<2>{local_rows(rows, comm.size(), comm.rank()), cols},
            stride_x);
        y_ = gko::experimental::distributed::Vector<etype>::create(
            exec, comm, gko::dim<2>{rows, cols},
            gko::dim<2>{local_rows(rows, comm.size(), comm.rank()), cols},
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

    void run() override { x_->compute_dot(gko::lend(y_), gko::lend(alpha_)); }

private:
    std::unique_ptr<gko::matrix::Dense<etype>> alpha_;
    std::unique_ptr<gko::experimental::distributed::Vector<etype>> x_;
    std::unique_ptr<gko::experimental::distributed::Vector<etype>> y_;
};


class NormOperation : public BenchmarkOperation {
public:
    NormOperation(std::shared_ptr<const gko::Executor> exec,
                  gko::mpi::communicator comm, gko::size_type rows,
                  gko::size_type cols, gko::size_type stride)
    {
        alpha_ = gko::matrix::Dense<etype>::create(exec, gko::dim<2>{1, cols});
        y_ = gko::experimental::distributed::Vector<etype>::create(
            exec, comm, gko::dim<2>{rows, cols},
            gko::dim<2>{local_rows(rows, comm.size(), comm.rank()), cols},
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

    void run() override { y_->compute_norm2(gko::lend(alpha_)); }

private:
    std::unique_ptr<gko::matrix::Dense<etype>> alpha_;
    std::unique_ptr<gko::experimental::distributed::Vector<etype>> y_;
};


struct dimensions {
    gko::size_type n;
    gko::size_type k;
    gko::size_type m;
    gko::size_type r;
    gko::size_type stride_x;
    gko::size_type stride_y;
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
    return result;
}


std::map<std::string, std::function<std::unique_ptr<BenchmarkOperation>(
                          std::shared_ptr<const gko::Executor>,
                          gko::mpi::communicator, dimensions)>>
    operation_map{
        {"copy",
         [](std::shared_ptr<const gko::Executor> exec,
            gko::mpi::communicator comm, dimensions dims) {
             return std::make_unique<CopyOperation>(
                 exec, comm, dims.n, dims.r, dims.stride_x, dims.stride_y);
         }},
        {"axpy",
         [](std::shared_ptr<const gko::Executor> exec,
            gko::mpi::communicator comm, dimensions dims) {
             return std::make_unique<AxpyOperation>(exec, comm, dims.n, dims.r,
                                                    dims.stride_x,
                                                    dims.stride_y, false);
         }},
        {"multiaxpy",
         [](std::shared_ptr<const gko::Executor> exec,
            gko::mpi::communicator comm, dimensions dims) {
             return std::make_unique<AxpyOperation>(exec, comm, dims.n, dims.r,
                                                    dims.stride_x,
                                                    dims.stride_y, true);
         }},
        {"scal",
         [](std::shared_ptr<const gko::Executor> exec,
            gko::mpi::communicator comm, dimensions dims) {
             return std::make_unique<ScalOperation>(exec, comm, dims.n, dims.r,
                                                    dims.stride_y, false);
         }},
        {"multiscal",
         [](std::shared_ptr<const gko::Executor> exec,
            gko::mpi::communicator comm, dimensions dims) {
             return std::make_unique<ScalOperation>(exec, comm, dims.n, dims.r,
                                                    dims.stride_y, true);
         }},
        {"dot",
         [](std::shared_ptr<const gko::Executor> exec,
            gko::mpi::communicator comm, dimensions dims) {
             return std::make_unique<DotOperation>(
                 exec, comm, dims.n, dims.r, dims.stride_x, dims.stride_y);
         }},
        {"norm", [](std::shared_ptr<const gko::Executor> exec,
                    gko::mpi::communicator comm, dimensions dims) {
             return std::make_unique<NormOperation>(exec, comm, dims.n, dims.r,
                                                    dims.stride_y);
         }}};


void apply_blas(const char* operation_name, std::shared_ptr<gko::Executor> exec,
                gko::mpi::communicator comm, rapidjson::Value& test_case,
                rapidjson::MemoryPoolAllocator<>& allocator)
{
    try {
        auto& blas_case = test_case["blas"];
        add_or_set_member(blas_case, operation_name,
                          rapidjson::Value(rapidjson::kObjectType), allocator);

        auto op =
            operation_map[operation_name](exec, comm, parse_dims(test_case));

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
        if (comm.rank() == 0) {
            std::cerr << "Error when processing test case " << test_case << "\n"
                      << "what(): " << e.what() << std::endl;
        }
    }
}


int main(int argc, char* argv[])
{
    gko::mpi::environment mpi_env{argc, argv};

    auto exec = executor_factory_mpi.at(FLAGS_executor)(MPI_COMM_WORLD);

    gko::mpi::communicator comm(MPI_COMM_WORLD);
    const auto rank = comm.rank();

    std::string header =
        "A benchmark for measuring performance of Ginkgo's BLAS-like "
        "operations.\nParameters for a benchmark case are:\n"
        "    n: number of rows for vectors output (required)\n"
        "    r: number of columns for vectors (optional, default 1)\n"
        "    stride: storage stride for both vectors (optional, default r)\n"
        "    stride_x: stride for input vector x (optional, default r)\n"
        "    stride_y: stride for in/out vector y (optional, default r)\n";
    std::string format = std::string() + "  [\n    { \"n\": 100 },\n" +
                         "    { \"n\": 200, \"r\": 20, \"stride_x\": 22 }\n" +
                         "  ]\n\n";
    initialize_argument_parsing(&argc, &argv, header, format);
    if (rank == 0) {
        std::string extra_information =
            "The operations are " + FLAGS_operations + "\n";
        print_general_information(extra_information);
    }

    auto operations = split(FLAGS_operations, ',');

    std::ifstream ifs(FLAGS_input);
    rapidjson::IStreamWrapper jcin(ifs);
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
            if (rank == 0) {
                std::clog << "Running test case: " << test_case << std::endl;
            }

            for (const auto& operation_name : operations) {
                apply_blas(operation_name.c_str(), exec, comm, test_case,
                           allocator);
                if (rank == 0) {
                    std::clog << "Current state:" << std::endl
                              << test_cases << std::endl;
                }
                backup_results(test_cases);
            }
        } catch (const std::exception& e) {
            if (rank == 0) {
                std::cerr << "Error setting up benchmark, what(): " << e.what()
                          << std::endl;
            }
        }
    }

    if (rank == 0) {
        std::cout << test_cases << std::endl;
    }
}
