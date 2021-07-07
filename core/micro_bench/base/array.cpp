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
        gen();
        exec = gko::ReferenceExecutor::create();
    }

    void TearDown(const ::benchmark::State &state) {}

    exec_ptr exec;
    std::uniform_int_distribution<> dis{1, 100};
    std::mt19937 gen;
};


BENCHMARK_F(Array, CreateWithoutExecutor)(benchmark::State &state)
{
    for (auto _ : state) {
        gko::Array<int> a;
    }
}


BENCHMARK_F(Array, CreateWithExecutor)(benchmark::State &state)
{
    for (auto _ : state) {
        gko::Array<int> a(this->exec);
    }
}


BENCHMARK_DEFINE_F(Array, CreateFromExistingData)(benchmark::State &state)
{
    for (auto _ : state) {
        gko::Array<int> a(this->exec, state.range(0), new int[state.range(0)],
                          std::default_delete<int[]>{});
    }
}

BENCHMARK_REGISTER_F(Array, CreateFromExistingData)
    ->RangeMultiplier(8)
    ->Range(64, 64 << 12);


BENCHMARK_DEFINE_F(Array, CreateView)(benchmark::State &state)
{
    auto data = new int[state.range(0)];
    for (auto _ : state) {
        auto view = gko::Array<int>::view(this->exec, state.range(0), data);
    }
    delete data;
}

BENCHMARK_REGISTER_F(Array, CreateView)
    ->RangeMultiplier(8)
    ->Range(64, 64 << 12);


BENCHMARK_DEFINE_F(Array, CreateFromDataOnExecutor)(benchmark::State &state)
{
    for (auto _ : state) {
        gko::Array<int> a{this->exec, state.range(0),
                          this->exec->template alloc<int>(state.range(0))};
    }
}

BENCHMARK_REGISTER_F(Array, CreateFromDataOnExecutor)
    ->RangeMultiplier(8)
    ->Range(64, 64 << 12);


BENCHMARK_DEFINE_F(Array, CreateFromRange)(benchmark::State &state)
{
    using std::begin;
    auto data = std::vector<int>{state.range(0), state.range(0) * 7};
    for (auto _ : state) {
        gko::Array<int> a{this->exec, begin(data), end(data)};
    }
}

BENCHMARK_REGISTER_F(Array, CreateFromRange)
    ->RangeMultiplier(8)
    ->Range(64, 64 << 12);


BENCHMARK_DEFINE_F(Array, CopyConstruct)(benchmark::State &state)
{
    auto x = gko::Array<int>{this->exec, state.range(0)};
    for (auto _ : state) {
        gko::Array<int> a(x);
    }
}

BENCHMARK_REGISTER_F(Array, CopyConstruct)
    ->RangeMultiplier(8)
    ->Range(64, 64 << 12);


BENCHMARK_DEFINE_F(Array, MoveConstruct)(benchmark::State &state)
{
    auto x = gko::Array<int>{this->exec, state.range(0)};
    for (auto _ : state) {
        gko::Array<int> a(std::move(x));
    }
}

BENCHMARK_REGISTER_F(Array, MoveConstruct)
    ->RangeMultiplier(8)
    ->Range(64, 64 << 12);


BENCHMARK_DEFINE_F(Array, CopyAssign)(benchmark::State &state)
{
    auto x1 = gko::Array<int>{this->exec, state.range(0)};
    auto x2 = gko::Array<int>{this->exec, state.range(0) + 2};
    for (auto _ : state) {
        x1 = x2;
    }
}

BENCHMARK_REGISTER_F(Array, CopyAssign)
    ->RangeMultiplier(8)
    ->Range(64, 64 << 12);


BENCHMARK_DEFINE_F(Array, MoveAssign)(benchmark::State &state)
{
    auto x1 = gko::Array<int>{this->exec, state.range(0)};
    auto x2 = gko::Array<int>{this->exec, state.range(0) + 3};
    for (auto _ : state) {
        x1 = std::move(x2);
    }
}

BENCHMARK_REGISTER_F(Array, MoveAssign)
    ->RangeMultiplier(8)
    ->Range(64, 64 << 12);


BENCHMARK_DEFINE_F(Array, CopyViews)(benchmark::State &state)
{
    auto data = new int[state.range(0)];
    auto data2 = new int[state.range(0)];
    auto view1 = gko::Array<int>::view(this->exec, state.range(0), data);
    auto view2 = gko::Array<int>::view(this->exec, state.range(0), data2);
    for (auto _ : state) {
        view1 = view2;
    }
    delete data, data2;
}

BENCHMARK_REGISTER_F(Array, CopyViews)->RangeMultiplier(8)->Range(64, 64 << 12);


BENCHMARK_DEFINE_F(Array, MoveViews)(benchmark::State &state)
{
    auto data = new int[state.range(0)];
    auto data2 = new int[state.range(0)];
    auto view1 = gko::Array<int>::view(this->exec, state.range(0), data);
    auto view2 = gko::Array<int>::view(this->exec, state.range(0), data2);
    for (auto _ : state) {
        view1 = std::move(view2);
    }
    delete data, data2;
}

BENCHMARK_REGISTER_F(Array, MoveViews)->RangeMultiplier(8)->Range(64, 64 << 12);


BENCHMARK_DEFINE_F(Array, TempClone)(benchmark::State &state)
{
    auto x = gko::Array<int>{this->exec, state.range(0)};
    for (auto _ : state) {
        auto tmp_clone = make_temporary_clone(this->exec, &x);
    }
}

BENCHMARK_REGISTER_F(Array, TempClone)->RangeMultiplier(8)->Range(64, 64 << 12);


BENCHMARK_DEFINE_F(Array, Clear)(benchmark::State &state)
{
    auto x = gko::Array<int>{this->exec, state.range(0)};
    for (auto _ : state) {
        x.clear();
    }
}

BENCHMARK_REGISTER_F(Array, Clear)->RangeMultiplier(8)->Range(64, 64 << 12);


BENCHMARK_DEFINE_F(Array, ResizeAndReset)(benchmark::State &state)
{
    auto x = gko::Array<int>{this->exec, state.range(0)};
    for (auto _ : state) {
        x.resize_and_reset(state.range(0) + this->dis(this->gen));
    }
}

BENCHMARK_REGISTER_F(Array, ResizeAndReset)
    ->RangeMultiplier(8)
    ->Range(64, 64 << 12);
