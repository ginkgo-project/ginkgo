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

#include <ginkgo/core/config/registry.hpp>


#include <functional>
#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/time.hpp>


#include "core/test/utils.hpp"


class Registry : public ::testing::Test {
protected:
    using Matrix = gko::matrix::Dense<float>;
    using Solver = gko::solver::Cg<float>;
    using Stop = gko::stop::Iteration;

    Registry()
        : exec{gko::ReferenceExecutor::create()},
          matrix{Matrix::create(exec)},
          solver_factory{Solver::build().on(exec)},
          stop_factory{Stop::build().on(exec)},
          func{[](const gko::config::pnode& config,
                  const gko::config::registry& context,
                  std::shared_ptr<const gko::Executor>& exec,
                  gko::config::type_descriptor td_for_child) {
              return gko::solver::Cg<float>::build().on(exec);
          }},
          reg{{{"func", func}}}
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Matrix> matrix;
    std::shared_ptr<typename Solver::Factory> solver_factory;
    std::shared_ptr<typename Stop::Factory> stop_factory;
    std::function<std::unique_ptr<gko::LinOpFactory>(
        const gko::config::pnode&, const gko::config::registry&,
        std::shared_ptr<const gko::Executor>&, gko::config::type_descriptor)>
        func;
    gko::config::registry reg;
};


TEST_F(Registry, InsertData)
{
    // Can put data
    ASSERT_TRUE(reg.emplace("matrix", matrix));
    ASSERT_TRUE(reg.emplace("solver_factory", solver_factory));
    ASSERT_TRUE(reg.emplace("stop_factory", stop_factory));
    // Do not insert the same key like normal map
    ASSERT_FALSE(reg.emplace("matrix", matrix));
    ASSERT_FALSE(reg.emplace("solver_factory", solver_factory));
    ASSERT_FALSE(reg.emplace("stop_factory", stop_factory));
}


TEST_F(Registry, SearchData)
{
    reg.emplace("matrix", matrix);
    reg.emplace("solver_factory", solver_factory);
    reg.emplace("stop_factory", stop_factory);

    auto found_matrix = reg.search_data<gko::LinOp>("matrix");
    auto found_solver_factory =
        reg.search_data<gko::LinOpFactory>("solver_factory");
    auto found_stop_factory =
        reg.search_data<gko::stop::CriterionFactory>("stop_factory");

    // get correct ptrs
    ASSERT_EQ(found_matrix, matrix);
    ASSERT_EQ(found_solver_factory, solver_factory);
    ASSERT_EQ(found_stop_factory, stop_factory);
    // get correct types
    ASSERT_TRUE((std::is_same<decltype(found_matrix),
                              std::shared_ptr<gko::LinOp>>::value));
    ASSERT_TRUE((std::is_same<decltype(found_solver_factory),
                              std::shared_ptr<gko::LinOpFactory>>::value));
    ASSERT_TRUE(
        (std::is_same<decltype(found_stop_factory),
                      std::shared_ptr<gko::stop::CriterionFactory>>::value));
}


TEST_F(Registry, SearchDataWithType)
{
    reg.emplace("matrix", matrix);
    reg.emplace("solver_factory", solver_factory);
    reg.emplace("stop_factory", stop_factory);

    auto found_matrix = reg.search_data<Matrix>("matrix");
    auto found_solver_factory =
        reg.search_data<Solver::Factory>("solver_factory");
    auto found_stop_factory = reg.search_data<Stop::Factory>("stop_factory");

    // get correct ptrs
    ASSERT_EQ(found_matrix, matrix);
    ASSERT_EQ(found_solver_factory, solver_factory);
    ASSERT_EQ(found_stop_factory, stop_factory);
    // get correct types
    ASSERT_TRUE(
        (std::is_same<decltype(found_matrix), std::shared_ptr<Matrix>>::value));
    ASSERT_TRUE(
        (std::is_same<decltype(found_solver_factory),
                      std::shared_ptr<typename Solver::Factory>>::value));
    ASSERT_TRUE((std::is_same<decltype(found_stop_factory),
                              std::shared_ptr<typename Stop::Factory>>::value));
}


TEST_F(Registry, ThrowIfNotFound)
{
    ASSERT_THROW(reg.search_data<gko::LinOp>("N"), std::out_of_range);
    ASSERT_THROW(reg.search_data<gko::LinOpFactory>("N"), std::out_of_range);
    ASSERT_THROW(reg.search_data<gko::stop::CriterionFactory>("N"),
                 std::out_of_range);
}


TEST_F(Registry, ThrowWithWrongType)
{
    reg.emplace("matrix", matrix);
    reg.emplace("solver_factory", solver_factory);
    reg.emplace("stop_factory", stop_factory);

    ASSERT_THROW(reg.search_data<gko::matrix::Dense<double>>("matrix"),
                 gko::NotSupported);
    ASSERT_THROW(
        reg.search_data<gko::solver::Cg<double>::Factory>("solver_factory"),
        gko::NotSupported);
    ASSERT_THROW(reg.search_data<gko::stop::Time::Factory>("stop_factory"),
                 gko::NotSupported);
}


TEST_F(Registry, GetBuildMap)
{
    auto factory = reg.get_build_map().at("func")(gko::config::pnode{"unused"},
                                                  reg, exec, {"", ""});

    ASSERT_NE(factory, nullptr);
}
