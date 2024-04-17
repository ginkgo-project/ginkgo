// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/config/registry.hpp>


#include <functional>
#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/time.hpp>


#include "core/config/config_helper.hpp"
#include "core/test/utils.hpp"


using namespace gko::config;


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
          func{[](const pnode& config, const registry& context,
                  type_descriptor td_for_child) {
              return gko::solver::Cg<float>::build();
          }},
          reg{{{"func", func}}}
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Matrix> matrix;
    std::shared_ptr<typename Solver::Factory> solver_factory;
    std::shared_ptr<typename Stop::Factory> stop_factory;
    std::function<gko::deferred_factory_parameter<gko::LinOpFactory>(
        const pnode&, const registry&, type_descriptor)>
        func;
    registry reg;
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
    auto factory =
        reg.get_build_map()
            .at("func")(pnode{"unused"}, reg, type_descriptor{"void", "void"})
            .on(exec);

    ASSERT_NE(factory, nullptr);
}


TEST(TypeDescriptor, TemplateCreate)
{
    auto td1 = make_type_descriptor<double, int>();
    auto td2 = make_type_descriptor<float>();
    auto td3 = make_type_descriptor<void, gko::int64>();
    auto td4 = make_type_descriptor<std::complex<float>, gko::int32>();
    auto td5 = make_type_descriptor<>();

    ASSERT_EQ(td1.get_value_typestr(), "double");
    ASSERT_EQ(td1.get_index_typestr(), "int");
    ASSERT_EQ(td2.get_value_typestr(), "float");
    ASSERT_EQ(td2.get_index_typestr(), "void");
    ASSERT_EQ(td3.get_value_typestr(), "void");
    ASSERT_EQ(td3.get_index_typestr(), "int64");
    ASSERT_EQ(td4.get_value_typestr(), "complex<float>");
    ASSERT_EQ(td4.get_index_typestr(), "int");
    ASSERT_EQ(td5.get_value_typestr(), "void");
    ASSERT_EQ(td5.get_index_typestr(), "void");
}


TEST(TypeDescriptor, Constructor)
{
    type_descriptor td1;
    type_descriptor td2("float");
    type_descriptor td3("double", "int");

    ASSERT_EQ(td1.get_value_typestr(), "void");
    ASSERT_EQ(td1.get_index_typestr(), "void");
    ASSERT_EQ(td2.get_value_typestr(), "float");
    ASSERT_EQ(td2.get_index_typestr(), "void");
    ASSERT_EQ(td3.get_value_typestr(), "double");
    ASSERT_EQ(td3.get_index_typestr(), "int");
}
