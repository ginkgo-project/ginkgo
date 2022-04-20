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

#include <ginkgo/core/base/array.hpp>


#include <cmath>
#include <random>
#include <type_traits>


#include <benchmark/benchmark.h>


#include <ginkgo/core/base/exception.hpp>


using exec_ptr = std::shared_ptr<gko::Executor>;


class Array : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State &state)
    {
        exec = gko::OmpExecutor::create();
    }

    void TearDown(const ::benchmark::State &state)
    {
        if (exec != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            exec->synchronize();
        }
    }

    exec_ptr exec;
    std::uniform_int_distribution<> dis{1, 100};
    std::mt19937 gen{};
};


BENCHMARK_F(Array, CreateWithoutExecutor)(benchmark::State &state)
{
    for (auto _ : state) {
        gko::Array<int> a;
    }
}


BENCHMARK_F(Array, CreateWithExecutor)(benchmark::State &state)
{
    auto cuda = gko::CudaExecutor::create(0, this->exec);
    for (auto _ : state) {
        gko::Array<int> a(cuda);
    }
}


BENCHMARK_DEFINE_F(Array, CreateFromDataOnExecutor)(benchmark::State &state)
{
    auto cuda = gko::CudaExecutor::create(0, this->exec);
    gko::size_type size = state.range(0);
    for (auto _ : state) {
        gko::Array<int> a{cuda, size, cuda->template alloc<int>(size)};
    }
}

BENCHMARK_REGISTER_F(Array, CreateFromDataOnExecutor)
    ->RangeMultiplier(8)
    ->Range(64, 64 << 10);


BENCHMARK_DEFINE_F(Array, CreateFromRange)(benchmark::State &state)
{
    using std::begin;
    auto cuda = gko::CudaExecutor::create(0, this->exec);
    auto size = static_cast<long>(state.range(0));
    auto data = std::vector<long>{size, size + 7};
    for (auto _ : state) {
        gko::Array<long> a{cuda, begin(data), end(data)};
    }
}

BENCHMARK_REGISTER_F(Array, CreateFromRange)
    ->RangeMultiplier(8)
    ->Range(64, 64 << 10);


BENCHMARK_DEFINE_F(Array, CopyConstructFromHost)(benchmark::State &state)
{
    gko::size_type size = state.range(0);
    auto cuda = gko::CudaExecutor::create(0, this->exec);
    auto x = gko::Array<int>{cuda->get_master(), size};
    for (auto _ : state) {
        gko::Array<int> a(cuda, x);
    }
}

BENCHMARK_REGISTER_F(Array, CopyConstructFromHost)
    ->RangeMultiplier(8)
    ->Range(64, 64 << 10);


BENCHMARK_DEFINE_F(Array, CopyConstructFromDevice)(benchmark::State &state)
{
    gko::size_type size = state.range(0);
    auto cuda = gko::CudaExecutor::create(0, this->exec);
    auto x = gko::Array<int>{cuda, size};
    for (auto _ : state) {
        gko::Array<int> a(x);
    }
}

BENCHMARK_REGISTER_F(Array, CopyConstructFromDevice)
    ->RangeMultiplier(8)
    ->Range(64, 64 << 10);


BENCHMARK_DEFINE_F(Array, MoveConstructFromHost)(benchmark::State &state)
{
    gko::size_type size = state.range(0);
    auto cuda = gko::CudaExecutor::create(0, this->exec);
    auto x = gko::Array<int>{cuda->get_master(), size};
    for (auto _ : state) {
        gko::Array<int> a(cuda, std::move(x));
    }
}

BENCHMARK_REGISTER_F(Array, MoveConstructFromHost)
    ->RangeMultiplier(8)
    ->Range(64, 64 << 10);


BENCHMARK_DEFINE_F(Array, MoveConstructFromDevice)(benchmark::State &state)
{
    gko::size_type size = state.range(0);
    auto cuda = gko::CudaExecutor::create(0, this->exec);
    auto x = gko::Array<int>{cuda, size};
    for (auto _ : state) {
        gko::Array<int> a(std::move(x));
    }
}

BENCHMARK_REGISTER_F(Array, MoveConstructFromDevice)
    ->RangeMultiplier(8)
    ->Range(64, 64 << 10);


BENCHMARK_DEFINE_F(Array, CopyAssignFromHost)(benchmark::State &state)
{
    gko::size_type size = state.range(0);
    auto cuda = gko::CudaExecutor::create(0, this->exec);
    auto x1 = gko::Array<int>{cuda, size};
    auto x2 = gko::Array<int>{cuda->get_master(), size + 2};
    for (auto _ : state) {
        x1 = x2;
    }
}

BENCHMARK_REGISTER_F(Array, CopyAssignFromHost)
    ->RangeMultiplier(8)
    ->Range(64, 64 << 10);


BENCHMARK_DEFINE_F(Array, CopyAssignFromDevice)(benchmark::State &state)
{
    gko::size_type size = state.range(0);
    auto cuda = gko::CudaExecutor::create(0, this->exec);
    auto x1 = gko::Array<int>{cuda, size};
    auto x2 = gko::Array<int>{cuda, size + 2};
    for (auto _ : state) {
        x1 = x2;
    }
}

BENCHMARK_REGISTER_F(Array, CopyAssignFromDevice)
    ->RangeMultiplier(8)
    ->Range(64, 64 << 10);


BENCHMARK_DEFINE_F(Array, MoveAssignFromHost)(benchmark::State &state)
{
    gko::size_type size = state.range(0);
    auto cuda = gko::CudaExecutor::create(0, this->exec);
    auto x1 = gko::Array<int>{cuda, size};
    auto x2 = gko::Array<int>{cuda->get_master(), size + 3};
    for (auto _ : state) {
        x1 = std::move(x2);
    }
}

BENCHMARK_REGISTER_F(Array, MoveAssignFromHost)
    ->RangeMultiplier(8)
    ->Range(64, 64 << 10);


BENCHMARK_DEFINE_F(Array, MoveAssignFromDevice)(benchmark::State &state)
{
    gko::size_type size = state.range(0);
    auto cuda = gko::CudaExecutor::create(0, this->exec);
    auto x1 = gko::Array<int>{cuda, size};
    auto x2 = gko::Array<int>{cuda, size + 3};
    for (auto _ : state) {
        x1 = std::move(x2);
    }
}

BENCHMARK_REGISTER_F(Array, MoveAssignFromDevice)
    ->RangeMultiplier(8)
    ->Range(64, 64 << 10);


BENCHMARK_DEFINE_F(Array, TempClone)(benchmark::State &state)
{
    gko::size_type size = state.range(0);
    auto cuda = gko::CudaExecutor::create(0, this->exec);
    auto x = gko::Array<int>{cuda->get_master(), size};
    for (auto _ : state) {
        auto tmp_clone = make_temporary_clone(cuda, &x);
    }
}

BENCHMARK_REGISTER_F(Array, TempClone)->RangeMultiplier(8)->Range(64, 64 << 10);


BENCHMARK_DEFINE_F(Array, Clear)(benchmark::State &state)
{
    gko::size_type size = state.range(0);
    auto cuda = gko::CudaExecutor::create(0, this->exec);
    auto x = gko::Array<int>{cuda, size};
    for (auto _ : state) {
        x.clear();
    }
}

BENCHMARK_REGISTER_F(Array, Clear)->RangeMultiplier(8)->Range(64, 64 << 10);


BENCHMARK_DEFINE_F(Array, ResizeAndReset)(benchmark::State &state)
{
    gko::size_type size = state.range(0);
    auto cuda = gko::CudaExecutor::create(0, this->exec);
    auto x = gko::Array<int>{cuda, size};
    for (auto _ : state) {
        x.resize_and_reset(size + this->dis(this->gen));
    }
}

BENCHMARK_REGISTER_F(Array, ResizeAndReset)
    ->RangeMultiplier(8)
    ->Range(64, 64 << 10);
