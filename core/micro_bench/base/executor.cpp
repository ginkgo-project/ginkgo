/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/base/executor.hpp>


#include <cmath>
#include <random>
#include <thread>
#include <type_traits>


#include <benchmark/benchmark.h>


#include <ginkgo/core/base/exception.hpp>


using exec_ptr = std::shared_ptr<gko::Executor>;


class ExampleOperation : public gko::Operation {
public:
    explicit ExampleOperation(int &val) : value(val) {}
    void run(std::shared_ptr<const gko::OmpExecutor>) const override
    {
        value = 1;
    }
    void run(std::shared_ptr<const gko::CudaExecutor>) const override
    {
        value = 2;
    }
    void run(std::shared_ptr<const gko::HipExecutor>) const override
    {
        value = 3;
    }
    void run(std::shared_ptr<const gko::DpcppExecutor>) const override
    {
        value = 4;
    }
    void run(std::shared_ptr<const gko::ReferenceExecutor>) const override
    {
        value = 5;
    }

    int &value;
};


template <typename ExecCreate>
static void executor_create(benchmark::State &st, ExecCreate &&exec_create)
{
    for (auto _ : st) {
        exec_create();
    }
}

BENCHMARK_CAPTURE(executor_create, Ref,
                  []() { auto exec = gko::ReferenceExecutor::create(); });
BENCHMARK_CAPTURE(executor_create, Omp,
                  []() { auto exec = gko::ReferenceExecutor::create(); });
BENCHMARK_CAPTURE(executor_create, Cuda, []() {
    auto exec = gko::CudaExecutor::create(0, gko::OmpExecutor::create());
});
BENCHMARK_CAPTURE(executor_create, Hip, []() {
    auto exec = gko::HipExecutor::create(0, gko::OmpExecutor::create());
});
BENCHMARK_CAPTURE(executor_create, Dpcpp, []() {
    auto exec = gko::DpcppExecutor::create(0, gko::OmpExecutor::create());
});


template <typename ExecCreate, typename ExecRun>
static void executor_run(benchmark::State &st, ExecCreate &&create,
                         ExecRun &&run)
{
    exec_ptr exec;
    create(exec);
    for (auto _ : st) {
        run(exec);
    }
}

BENCHMARK_CAPTURE(
    executor_run, Ref,
    [](exec_ptr &exec) { exec = gko::ReferenceExecutor::create(); },
    [](exec_ptr &exec) {
        int value = 0;
        exec->run(ExampleOperation(value));
    });
BENCHMARK_CAPTURE(
    executor_run, Omp,
    [](exec_ptr &exec) { exec = gko::OmpExecutor::create(); },
    [](exec_ptr &exec) {
        int value = 0;
        exec->run(ExampleOperation(value));
    });
BENCHMARK_CAPTURE(
    executor_run, Cuda,
    [](exec_ptr &exec) {
        exec = gko::CudaExecutor::create(0, gko::OmpExecutor::create());
    },
    [](exec_ptr &exec) {
        int value = 0;
        exec->run(ExampleOperation(value));
    });
BENCHMARK_CAPTURE(
    executor_run, Hip,
    [](exec_ptr &exec) {
        exec = gko::HipExecutor::create(0, gko::OmpExecutor::create());
    },
    [](exec_ptr &exec) {
        int value = 0;
        exec->run(ExampleOperation(value));
    });
BENCHMARK_CAPTURE(
    executor_run, Dpcpp,
    [](exec_ptr &exec) {
        exec = gko::DpcppExecutor::create(0, gko::OmpExecutor::create());
    },
    [](exec_ptr &exec) {
        int value = 0;
        exec->run(ExampleOperation(value));
    });
