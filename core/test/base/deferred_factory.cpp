// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>


template <typename T>
using dfp = gko::deferred_factory_parameter<T>;


// Note: the following Factory structure is not identical to Ginkgo Factory
// structure, but it is easier setup without too many dependencies and
// inheritances.
struct DummyBaseFactory {
    virtual ~DummyBaseFactory() = default;
    struct param {
        std::unique_ptr<DummyBaseFactory> on(
            std::shared_ptr<const gko::Executor>) const
        {
            return std::make_unique<DummyBaseFactory>();
        }
    };
};


struct DummyFactory : DummyBaseFactory {
    struct param {
        std::unique_ptr<DummyFactory> on(
            std::shared_ptr<const gko::Executor>) const
        {
            return std::make_unique<DummyFactory>();
        }
    };
};


struct DummyFactory2 : DummyBaseFactory {
    struct param : public gko::enable_parameters_type<param, DummyFactory2> {
        using parameters_type = param;
        std::vector<std::shared_ptr<const DummyBaseFactory>>
            GKO_DEFERRED_FACTORY_VECTOR_PARAMETER(const_factory_list);

        std::vector<std::shared_ptr<DummyBaseFactory>>
            GKO_DEFERRED_FACTORY_VECTOR_PARAMETER(factory_list);
    };

    const param& get_parameters() const noexcept { return parameters_; };

    void add_logger(std::shared_ptr<const gko::log::Logger> logger) {}

    DummyFactory2(std::shared_ptr<const gko::Executor>, const param& parameters)
        : parameters_(parameters)
    {}

private:
    param parameters_;
};


using DF = DummyFactory;
using DBF = DummyBaseFactory;
using DF2 = DummyFactory2;


// used to distinguish specialization for function check
struct DummyFlag {};


// test_impl is to check the constructor available or not in the compile time.
// Note. It only checks the signature with the template and SFINAE. If the
// compilation error is only in the function/constructor definition, it will
// still give the true back.
template <typename, typename T, typename...>
struct test_impl : std::false_type {};

// specialization for constructor
template <typename T, typename... Args>
struct test_impl<gko::xstd::void_t<decltype(T(std::declval<Args>()...))>, T,
                 Args...> : std::true_type {};

// specialization for DF2 with_factory_list
template <typename... Args>
struct test_impl<gko::xstd::void_t<decltype(
                     DF2::param{}.with_factory_list(std::declval<Args>()...))>,
                 DummyFlag, Args...> : std::true_type {};

// test the object can be constructable or not with Args.
template <typename T, typename... Args>
using test = test_impl<void, T, Args...>;

// test the DF2::param{}.with_factory_list can be called or not with Args.
template <typename... Args>
using test_with_factory = test_impl<void, DummyFlag, Args...>;


class DeferredFactoryParameter : public ::testing::Test {
protected:
    DF::param df_param{};
    DBF::param dbf_param{};
    const DF::param const_df_param{};
    const DBF::param const_dbf_param{};
    std::shared_ptr<DF> shared_df = std::make_shared<DF>();
    std::shared_ptr<DBF> shared_dbf = std::make_shared<DBF>();
    std::shared_ptr<const DF> shared_const_df = std::make_shared<DF>();
    std::shared_ptr<const DBF> shared_const_dbf = std::make_shared<DBF>();
    dfp<DF> dfp_df{std::make_shared<DF>()};
    dfp<DBF> dfp_dbf{std::make_shared<DBF>()};
    dfp<const DF> dfp_const_df{std::make_shared<DF>()};
    dfp<const DBF> dfp_const_dbf{std::make_shared<DBF>()};
    dfp<DF> nest_dfp_df{dfp_df};
    dfp<DBF> nest_dfp_dbf{dfp_dbf};
    dfp<const DF> nest_dfp_const_df{dfp_const_df};
    dfp<const DBF> nest_dfp_const_dbf{dfp_const_dbf};
};


TEST_F(DeferredFactoryParameter, CanBeDefaultConstructed)
{
    auto fact = dfp<DBF>();
    auto fact2 = dfp<const DBF>();

    ASSERT_TRUE(fact.is_empty());
    ASSERT_THROW(fact.on(nullptr), gko::NotSupported);
    ASSERT_TRUE(fact2.is_empty());
    ASSERT_THROW(fact2.on(nullptr), gko::NotSupported);
}


TEST_F(DeferredFactoryParameter, CanBeConstructedFromNullptr)
{
    auto fact = dfp<DBF>(nullptr);
    auto fact2 = dfp<const DBF>(nullptr);

    ASSERT_FALSE(fact.is_empty());
    ASSERT_EQ(fact.on(nullptr), nullptr);
    ASSERT_FALSE(fact2.is_empty());
    ASSERT_EQ(fact2.on(nullptr), nullptr);
}


TEST_F(DeferredFactoryParameter, CheckNonConstConstructor)
{
    // Itself
    // shared_ptr
    auto fact0 = dfp<DBF>(this->shared_dbf);
    // unique_ptr
    auto fact1 = dfp<DBF>(this->dbf_param.on(nullptr));
    // const param
    auto fact2 = dfp<DBF>(this->const_dbf_param);
    // param
    auto fact3 = dfp<DBF>(this->dbf_param);
    // deferred_factory_parameter
    auto fact4 = dfp<DBF>(this->dfp_dbf);
    // Childtype
    auto fact5 = dfp<DBF>(this->shared_df);
    auto fact6 = dfp<DBF>(this->df_param.on(nullptr));
    auto fact7 = dfp<DBF>(this->const_df_param);
    auto fact8 = dfp<DBF>(this->df_param);
    auto fact9 = dfp<DBF>(this->dfp_df);

    for (auto& fact : {fact0, fact1, fact2, fact3, fact4}) {
        ASSERT_TRUE(std::dynamic_pointer_cast<DBF>(fact.on(nullptr)));
    }
    for (auto& fact : {fact5, fact6, fact7, fact8, fact9}) {
        ASSERT_TRUE(std::dynamic_pointer_cast<DF>(fact.on(nullptr)));
    }
}


TEST_F(DeferredFactoryParameter, CheckConstConstructor)
{
    // Itself
    // shared_ptr
    auto fact0 = dfp<const DBF>(this->shared_dbf);
    // shared_ptr const
    auto fact1 = dfp<const DBF>(this->shared_const_dbf);
    // unique_ptr
    auto fact2 = dfp<const DBF>(this->dbf_param.on(nullptr));
    // unique_ptr const
    auto fact3 = dfp<const DBF>(
        static_cast<std::unique_ptr<const DBF>>(this->dbf_param.on(nullptr)));
    // const param
    auto fact4 = dfp<const DBF>(this->const_dbf_param);
    // param
    auto fact5 = dfp<const DBF>(this->dbf_param);
    // deferred_factory_parameter
    auto fact6 = dfp<const DBF>(this->dfp_dbf);
    // deferred_factory_parameter const
    auto fact7 = dfp<const DBF>(this->dfp_const_dbf);
    // Childtype
    auto fact_child0 = dfp<const DBF>(this->shared_df);
    auto fact_child1 = dfp<const DBF>(this->shared_const_df);
    auto fact_child2 = dfp<const DBF>(this->df_param.on(nullptr));
    auto fact_child3 = dfp<const DBF>(
        static_cast<std::unique_ptr<const DF>>(this->df_param.on(nullptr)));
    auto fact_child4 = dfp<const DBF>(this->const_df_param);
    auto fact_child5 = dfp<const DBF>(this->df_param);
    auto fact_child6 = dfp<const DBF>(this->dfp_df);
    auto fact_child7 = dfp<const DBF>(this->dfp_const_df);

    for (auto& fact :
         {fact0, fact1, fact2, fact3, fact4, fact5, fact6, fact7}) {
        ASSERT_TRUE(std::dynamic_pointer_cast<const DBF>(fact.on(nullptr)));
    }
    for (auto& fact : {fact_child0, fact_child1, fact_child2, fact_child3,
                       fact_child4, fact_child5, fact_child6, fact_child7}) {
        ASSERT_TRUE(std::dynamic_pointer_cast<const DF>(fact.on(nullptr)));
    }
}


TEST_F(DeferredFactoryParameter, ValidateNotAllowedFromNonConstConstructor)
{
    ASSERT_TRUE((test<dfp<DBF>, std::shared_ptr<DBF>>::value));
    // The following can not be constructed. Using the corresponding constructor
    // leads to a compile-time error.
    ASSERT_FALSE((test<dfp<DBF>, std::shared_ptr<const DBF>>::value));
    ASSERT_FALSE((test<dfp<DBF>, std::unique_ptr<const DBF>>::value));
    ASSERT_FALSE((test<dfp<DBF>, dfp<const DBF>>::value));
    ASSERT_FALSE((test<dfp<DF>, dfp<DBF>>::value));
    ASSERT_FALSE((test<dfp<DF>, std::shared_ptr<DBF>>::value));
    ASSERT_FALSE((test<dfp<DF>, dfp<DF2>>::value));
    ASSERT_FALSE((test<dfp<DF>, std::shared_ptr<DF2>>::value));
}


TEST_F(DeferredFactoryParameter, CheckMacroWithConstList)
{
    auto result =
        DummyFactory2::param{}
            .with_const_factory_list(
                this->df_param, this->const_df_param, this->shared_df,
                this->shared_const_df, this->df_param.on(nullptr), this->dfp_df,
                this->dfp_const_df, this->nest_dfp_df, this->nest_dfp_const_df)
            .on(nullptr);
    auto result_base =
        DummyFactory2::param{}
            .with_const_factory_list(this->dbf_param, this->const_dbf_param,
                                     this->shared_dbf, this->shared_const_dbf,
                                     this->dbf_param.on(nullptr), this->dfp_dbf,
                                     this->dfp_const_dbf, this->nest_dfp_dbf,
                                     this->nest_dfp_const_dbf)
            .on(nullptr);

    auto& factory_list = result->get_parameters().const_factory_list;
    auto& base_factory_list = result_base->get_parameters().const_factory_list;
    const auto num = factory_list.size();
    ASSERT_EQ(num, 9);
    ASSERT_EQ(base_factory_list.size(), 9);
    for (int i = 0; i < num; i++) {
        // The list requires const DummyBaseFactory, so they must be const
        ASSERT_TRUE(std::dynamic_pointer_cast<const DF>(factory_list.at(i)));
        ASSERT_TRUE(
            std::dynamic_pointer_cast<const DBF>(base_factory_list.at(i)));
    }
}


TEST_F(DeferredFactoryParameter, CheckMacroWithNonConstList)
{
    auto result =
        DummyFactory2::param{}
            .with_factory_list(this->df_param, this->const_df_param,
                               this->shared_df, this->df_param.on(nullptr),
                               this->dfp_df, this->nest_dfp_df)
            .on(nullptr);
    auto result_base =
        DummyFactory2::param{}
            .with_factory_list(this->dbf_param, this->const_dbf_param,
                               this->shared_dbf, this->dbf_param.on(nullptr),
                               this->dfp_dbf, this->nest_dfp_dbf)
            .on(nullptr);

    auto& factory_list = result->get_parameters().factory_list;
    auto& base_factory_list = result_base->get_parameters().factory_list;
    const auto num = factory_list.size();
    ASSERT_EQ(num, 6);
    ASSERT_EQ(base_factory_list.size(), 6);
    for (int i = 0; i < num; i++) {
        // The list requires DummyBaseFactory, so they must be non-const
        ASSERT_TRUE(std::dynamic_pointer_cast<DF>(factory_list.at(i)));
        ASSERT_TRUE(std::dynamic_pointer_cast<DBF>(base_factory_list.at(i)));
    }
}


TEST_F(DeferredFactoryParameter, CheckMacroWithConstVector)
{
    auto const_dbf_vec = std::vector<std::shared_ptr<const DBF>>{
        this->shared_dbf, this->shared_dbf};
    auto dbf_vec =
        std::vector<std::shared_ptr<DBF>>{this->shared_dbf, this->shared_dbf};
    auto dfp_const_dbf_vec =
        std::vector<dfp<const DBF>>{this->dbf_param, this->shared_dbf};
    auto dfp_dbf_vec = std::vector<dfp<DBF>>{this->dbf_param, this->shared_dbf};
    auto dbf_param_vec =
        std::vector<DBF::param>{this->dbf_param, this->dbf_param};
    // child
    auto const_df_vec = std::vector<std::shared_ptr<const DF>>{this->shared_df,
                                                               this->shared_df};
    auto df_vec =
        std::vector<std::shared_ptr<DF>>{this->shared_df, this->shared_df};
    auto dfp_const_df_vec =
        std::vector<dfp<const DF>>{this->df_param, this->shared_df};
    auto dfp_df_vec = std::vector<dfp<DF>>{this->df_param, this->shared_df};
    auto df_param_vec = std::vector<DF::param>{this->df_param, this->df_param};
    std::vector<std::shared_ptr<DF2>> result_base_vector;
    std::vector<std::shared_ptr<DF2>> result_vector;

    result_base_vector.emplace_back(
        DF2::param{}.with_const_factory_list(const_dbf_vec).on(nullptr));
    result_base_vector.emplace_back(
        DF2::param{}.with_const_factory_list(dbf_vec).on(nullptr));
    result_base_vector.emplace_back(
        DF2::param{}.with_const_factory_list(dfp_const_dbf_vec).on(nullptr));
    result_base_vector.emplace_back(
        DF2::param{}.with_const_factory_list(dfp_dbf_vec).on(nullptr));
    result_base_vector.emplace_back(
        DF2::param{}.with_const_factory_list(dbf_param_vec).on(nullptr));
    // For child input
    result_vector.emplace_back(
        DF2::param{}.with_const_factory_list(const_df_vec).on(nullptr));
    result_vector.emplace_back(
        DF2::param{}.with_const_factory_list(df_vec).on(nullptr));
    result_vector.emplace_back(
        DF2::param{}.with_const_factory_list(dfp_const_df_vec).on(nullptr));
    result_vector.emplace_back(
        DF2::param{}.with_const_factory_list(dfp_df_vec).on(nullptr));
    result_vector.emplace_back(
        DF2::param{}.with_const_factory_list(df_param_vec).on(nullptr));

    for (const auto& result : result_base_vector) {
        auto& factory_list = result->get_parameters().const_factory_list;
        ASSERT_TRUE(std::dynamic_pointer_cast<const DBF>(factory_list.at(0)));
        ASSERT_TRUE(std::dynamic_pointer_cast<const DBF>(factory_list.at(1)));
    }
    for (const auto& result : result_vector) {
        auto& factory_list = result->get_parameters().const_factory_list;
        ASSERT_TRUE(std::dynamic_pointer_cast<const DF>(factory_list.at(0)));
        ASSERT_TRUE(std::dynamic_pointer_cast<const DF>(factory_list.at(1)));
    }
}


TEST_F(DeferredFactoryParameter, CheckMacroWithNonConstVector)
{
    auto dbf_vec =
        std::vector<std::shared_ptr<DBF>>{this->shared_dbf, this->shared_dbf};
    auto dfp_dbf_vec = std::vector<dfp<DBF>>{this->dbf_param, this->shared_dbf};
    auto dbf_param_vec =
        std::vector<DBF::param>{this->dbf_param, this->dbf_param};
    // child
    auto df_vec =
        std::vector<std::shared_ptr<DF>>{this->shared_df, this->shared_df};
    auto dfp_df_vec = std::vector<dfp<DF>>{this->df_param, this->shared_df};
    auto df_param_vec = std::vector<DF::param>{this->df_param, this->df_param};
    std::vector<std::shared_ptr<DF2>> result_base_vector;
    std::vector<std::shared_ptr<DF2>> result_vector;

    result_base_vector.emplace_back(
        DF2::param{}.with_factory_list(dbf_vec).on(nullptr));
    result_base_vector.emplace_back(
        DF2::param{}.with_factory_list(dfp_dbf_vec).on(nullptr));
    result_base_vector.emplace_back(
        DF2::param{}.with_factory_list(dbf_param_vec).on(nullptr));
    // For child input
    result_vector.emplace_back(
        DF2::param{}.with_factory_list(df_vec).on(nullptr));
    result_vector.emplace_back(
        DF2::param{}.with_factory_list(dfp_df_vec).on(nullptr));
    result_vector.emplace_back(
        DF2::param{}.with_factory_list(df_param_vec).on(nullptr));

    for (const auto& result : result_base_vector) {
        auto& factory_list = result->get_parameters().factory_list;
        ASSERT_TRUE(std::dynamic_pointer_cast<DBF>(factory_list.at(0)));
        ASSERT_TRUE(std::dynamic_pointer_cast<DBF>(factory_list.at(1)));
    }
    for (const auto& result : result_vector) {
        auto& factory_list = result->get_parameters().factory_list;
        ASSERT_TRUE(std::dynamic_pointer_cast<DF>(factory_list.at(0)));
        ASSERT_TRUE(std::dynamic_pointer_cast<DF>(factory_list.at(1)));
    }
}


TEST_F(DeferredFactoryParameter, ValidateNotAllowedFromMacroWithNonConst)
{
    ASSERT_TRUE((test_with_factory<std::vector<std::shared_ptr<DBF>>>::value));
    ASSERT_TRUE(
        (test_with_factory<std::shared_ptr<DBF>, std::shared_ptr<DF>>::value));
    ASSERT_FALSE(
        (test_with_factory<std::vector<std::shared_ptr<const DBF>>>::value));
    ASSERT_FALSE((test_with_factory<std::vector<dfp<const DBF>>>::value));
    ASSERT_FALSE(
        (test_with_factory<std::vector<std::shared_ptr<DummyFlag>>>::value));
    ASSERT_FALSE((test_with_factory<std::shared_ptr<const DBF>,
                                    std::shared_ptr<const DF>>::value));
}
