// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <chrono>
#include <complex>
#include <iomanip>
#include <limits>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/property_tree.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/config/type_descriptor.hpp>
#include <ginkgo/core/config/write_config.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/idr.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/core/stop/time.hpp>

#include "core/config/config_helper.hpp"
#include "core/config/write_helper.hpp"
#include "core/test/utils.hpp"


namespace gko {
namespace config {


// Lets gtest print a pnode as JSON when an assertion fails.
void PrintTo(const pnode& node, std::ostream* os)
{
    switch (node.get_tag()) {
    case pnode::tag_t::empty:
        *os << "null";
        break;
    case pnode::tag_t::boolean:
        *os << (node.get_boolean() ? "true" : "false");
        break;
    case pnode::tag_t::integer:
        *os << node.get_integer();
        break;
    case pnode::tag_t::real: {
        const auto precision =
            os->precision(std::numeric_limits<double>::max_digits10);
        *os << node.get_real();
        os->precision(precision);
        break;
    }
    case pnode::tag_t::string:
        *os << '"' << node.get_string() << '"';
        break;
    case pnode::tag_t::array: {
        *os << '[';
        auto separator = "";
        for (const auto& item : node.get_array()) {
            *os << separator;
            PrintTo(item, os);
            separator = ", ";
        }
        *os << ']';
        break;
    }
    case pnode::tag_t::map: {
        *os << '{';
        auto separator = "";
        for (const auto& item : node.get_map()) {
            *os << separator << '"' << item.first << "\": ";
            PrintTo(item.second, os);
            separator = ", ";
        }
        *os << '}';
        break;
    }
    }
}


}  // namespace config
}  // namespace gko


namespace {


using namespace gko::config;


using criterion_vector =
    std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>;


// keys that parse reads but write_config writes separately or not at all
const std::set<std::string> not_written_by_default{
    "type",      "value_type",    "index_type", "generated_preconditioner",
    "nullspace", "left_nullspace"};


/**
 * One LinOp type for the typed tests, with a factory that sets every
 * parameter to a non-default value.
 */
template <typename T>
struct write_case;


template <typename ValueType>
struct write_case<gko::solver::Cg<ValueType>> {
    static constexpr const char* type = "solver::Cg";
    using index_type = gko::int32;

    static std::unique_ptr<gko::LinOpFactory> non_default(
        std::shared_ptr<const gko::Executor> exec)
    {
        return gko::solver::Cg<ValueType>::build()
            .with_criteria(gko::stop::combine(criterion_vector{
                gko::stop::Iteration::build().with_max_iters(7u).on(exec),
                gko::stop::ResidualNorm<ValueType>::build()
                    .with_reduction_factor(1e-5)
                    .on(exec)}))
            .with_preconditioner(
                gko::preconditioner::Jacobi<ValueType, index_type>::build()
                    .with_max_block_size(16u))
            .on(exec);
    }
};


template <typename ValueType, typename IndexType>
struct write_case<gko::preconditioner::Jacobi<ValueType, IndexType>> {
    static constexpr const char* type = "preconditioner::Jacobi";
    using index_type = IndexType;

    static std::unique_ptr<gko::LinOpFactory> non_default(
        std::shared_ptr<const gko::Executor> exec)
    {
        return gko::preconditioner::Jacobi<ValueType, IndexType>::build()
            .with_max_block_size(16u)
            .with_max_block_stride(32u)
            .with_skip_sorting(true)
            .with_storage_optimization(gko::precision_reduction(0, 1))
            .with_accuracy(gko::remove_complex<ValueType>{1e-2})
            .with_aggregate_l1(true)
            .on(exec);
    }
};


template <typename T>
class WriteConfig : public ::testing::Test {
protected:
    using linop_type = T;
    using write_case = ::write_case<T>;

    WriteConfig() : exec(gko::ReferenceExecutor::create()) {}

    // The value type comes from the written tree, the index type from here.
    type_descriptor td() const
    {
        return make_type_descriptor<double, typename write_case::index_type>();
    }

    void assert_round_trip(const gko::LinOpFactory& factory)
    {
        auto written = write_config(factory);

        auto parsed = parse(written, registry{}, td()).on(exec);

        ASSERT_NE(
            dynamic_cast<const typename linop_type::Factory*>(parsed.get()),
            nullptr);
        ASSERT_EQ(write_config(*parsed), written);
    }

    std::shared_ptr<const gko::Executor> exec;
};

using WriteConfigTypes =
    ::testing::Types<gko::solver::Cg<double>, gko::solver::Cg<float>,
                     gko::solver::Cg<std::complex<float>>,
                     gko::preconditioner::Jacobi<double, gko::int32>,
                     gko::preconditioner::Jacobi<float, gko::int64>>;

TYPED_TEST_SUITE(WriteConfig, WriteConfigTypes, TypenameNameGenerator);


TYPED_TEST(WriteConfig, RoundTrips)
{
    using write_case = typename TestFixture::write_case;

    this->assert_round_trip(*write_case::non_default(this->exec));
}


TYPED_TEST(WriteConfig, RoundTripsDefaults)
{
    using linop_type = typename TestFixture::linop_type;

    this->assert_round_trip(*linop_type::build().on(this->exec));
}


TYPED_TEST(WriteConfig, WritesTypeAndValueType)
{
    using linop_type = typename TestFixture::linop_type;
    using value_type = typename linop_type::value_type;
    using write_case = typename TestFixture::write_case;

    auto written = write_config(*linop_type::build().on(this->exec));

    ASSERT_EQ(written.get("type"), pnode{write_case::type});
    ASSERT_EQ(written.get("value_type"), pnode{type_string<value_type>::str()});
    ASSERT_FALSE(written.get("index_type"));
}


TYPED_TEST(WriteConfig, WritesEveryParsedKey)
{
    using linop_type = typename TestFixture::linop_type;
    using write_case = typename TestFixture::write_case;
    std::set<std::string> parsed_keys;
    {
        record_requested_keys recorder;
        parse(pnode{{{"type", pnode{write_case::type}}}}, registry{},
              this->td());
        parsed_keys = recorder.keys();
    }

    auto written = write_config(*linop_type::build().on(this->exec));

    ASSERT_FALSE(parsed_keys.empty());
    for (const auto& key : parsed_keys) {
        if (not_written_by_default.count(key) == 0) {
            EXPECT_EQ(written.get_map().count(key), 1)
                << "write_config does not write the key " << key;
        }
    }
}


TYPED_TEST(WriteConfig, WritesTheValueOfEveryParsedKey)
{
    using linop_type = typename TestFixture::linop_type;
    using write_case = typename TestFixture::write_case;
    std::set<std::string> parsed_keys;
    {
        record_requested_keys recorder;
        parse(pnode{{{"type", pnode{write_case::type}}}}, registry{},
              this->td());
        parsed_keys = recorder.keys();
    }

    auto written_default = write_config(*linop_type::build().on(this->exec));
    auto written = write_config(*write_case::non_default(this->exec));

    // non_default changes every parameter, so every written key differs
    for (const auto& key : parsed_keys) {
        if (not_written_by_default.count(key) == 0) {
            EXPECT_NE(written.get(key), written_default.get(key))
                << "write_config does not write the value of the key " << key;
        }
    }
}


template <typename T>
struct criterion_case;


template <>
struct criterion_case<gko::stop::Iteration> {
    static constexpr const char* type = "Iteration";

    static std::shared_ptr<const gko::stop::CriterionFactory> non_default(
        std::shared_ptr<const gko::Executor> exec)
    {
        return gko::stop::Iteration::build().with_max_iters(7u).on(exec);
    }
};


template <>
struct criterion_case<gko::stop::Time> {
    static constexpr const char* type = "Time";

    static std::shared_ptr<const gko::stop::CriterionFactory> non_default(
        std::shared_ptr<const gko::Executor> exec)
    {
        return gko::stop::Time::build()
            .with_time_limit(std::chrono::milliseconds{5})
            .on(exec);
    }
};


template <typename ValueType>
struct criterion_case<gko::stop::ResidualNorm<ValueType>> {
    static constexpr const char* type = "ResidualNorm";

    static std::shared_ptr<const gko::stop::CriterionFactory> non_default(
        std::shared_ptr<const gko::Executor> exec)
    {
        return gko::stop::ResidualNorm<ValueType>::build()
            .with_reduction_factor(1e-5)
            .with_baseline(gko::stop::mode::initial_resnorm)
            .on(exec);
    }
};


template <typename ValueType>
struct criterion_case<gko::stop::ImplicitResidualNorm<ValueType>> {
    static constexpr const char* type = "ImplicitResidualNorm";

    static std::shared_ptr<const gko::stop::CriterionFactory> non_default(
        std::shared_ptr<const gko::Executor> exec)
    {
        return gko::stop::ImplicitResidualNorm<ValueType>::build()
            .with_reduction_factor(1e-4)
            .with_baseline(gko::stop::mode::absolute)
            .on(exec);
    }
};


template <typename T>
class WriteCriterion : public ::testing::Test {
protected:
    using criterion_type = T;
    using criterion_case = ::criterion_case<T>;

    WriteCriterion() : exec(gko::ReferenceExecutor::create()) {}

    void assert_round_trip(const gko::stop::CriterionFactory& factory)
    {
        auto written = write_config(factory);

        auto parsed = parse_or_get_factory<const gko::stop::CriterionFactory>(
                          written, registry{}, make_type_descriptor<>())
                          .on(exec);

        ASSERT_NE(
            dynamic_cast<const typename criterion_type::Factory*>(parsed.get()),
            nullptr);
        ASSERT_EQ(write_config(*parsed), written);
    }

    std::shared_ptr<const gko::Executor> exec;
};

using WriteCriterionTypes = ::testing::Types<
    gko::stop::Iteration, gko::stop::Time, gko::stop::ResidualNorm<double>,
    gko::stop::ResidualNorm<float>, gko::stop::ImplicitResidualNorm<double>,
    gko::stop::ImplicitResidualNorm<std::complex<float>>>;

TYPED_TEST_SUITE(WriteCriterion, WriteCriterionTypes, TypenameNameGenerator);


TYPED_TEST(WriteCriterion, RoundTrips)
{
    using criterion_case = typename TestFixture::criterion_case;

    this->assert_round_trip(*criterion_case::non_default(this->exec));
}


TYPED_TEST(WriteCriterion, RoundTripsDefaults)
{
    using criterion_type = typename TestFixture::criterion_type;

    this->assert_round_trip(*criterion_type::build().on(this->exec));
}


TYPED_TEST(WriteCriterion, WritesEveryParsedKey)
{
    using criterion_type = typename TestFixture::criterion_type;
    using criterion_case = typename TestFixture::criterion_case;
    std::set<std::string> parsed_keys;
    {
        record_requested_keys recorder;
        parse_or_get_factory<const gko::stop::CriterionFactory>(
            pnode{{{"type", pnode{criterion_case::type}}}}, registry{},
            make_type_descriptor<>());
        parsed_keys = recorder.keys();
    }

    auto written = write_config(*criterion_type::build().on(this->exec));
    auto written_non_default =
        write_config(*criterion_case::non_default(this->exec));

    ASSERT_EQ(written.get("type"), pnode{criterion_case::type});
    ASSERT_FALSE(parsed_keys.empty());
    for (const auto& key : parsed_keys) {
        if (not_written_by_default.count(key) == 0) {
            EXPECT_EQ(written.get_map().count(key), 1)
                << "write_config does not write the key " << key;
            EXPECT_NE(written_non_default.get(key), written.get(key))
                << "write_config does not write the value of the key " << key;
        }
    }
}


class WriteConfigExamples : public ::testing::Test {
protected:
    WriteConfigExamples() : exec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::Executor> exec;
};


TEST_F(WriteConfigExamples, WritesCgWithJacobiAndCombinedCriteria)
{
    auto factory =
        gko::solver::Cg<double>::build()
            .with_criteria(gko::stop::combine(criterion_vector{
                gko::stop::Iteration::build().with_max_iters(7u).on(exec),
                gko::stop::ResidualNorm<double>::build()
                    .with_reduction_factor(1e-5)
                    .on(exec)}))
            .with_preconditioner(
                gko::preconditioner::Jacobi<double, gko::int32>::build()
                    .with_max_block_size(1u))
            .on(exec);

    auto written = write_config(*factory);

    pnode expected{
        {{"type", pnode{"solver::Cg"}},
         {"value_type", pnode{"float64"}},
         {"criteria",
          pnode{pnode::array_type{
              pnode{{{"type", pnode{"Iteration"}}, {"max_iters", pnode{7}}}},
              pnode{{{"type", pnode{"ResidualNorm"}},
                     {"value_type", pnode{"float64"}},
                     {"reduction_factor", pnode{1e-5}},
                     {"baseline", pnode{"rhs_norm"}}}}}}},
         {"preconditioner",
          pnode{{{"type", pnode{"preconditioner::Jacobi"}},
                 {"value_type", pnode{"float64"}},
                 {"max_block_size", pnode{1}},
                 {"max_block_stride", pnode{0}},
                 {"skip_sorting", pnode{false}},
                 {"storage_optimization",
                  pnode{pnode::array_type{pnode{0}, pnode{0}}}},
                 {"accuracy", pnode{1e-1}},
                 {"aggregate_l1", pnode{false}}}}}}};
    ASSERT_EQ(written, expected);
}


TEST_F(WriteConfigExamples, WritesCriteriaAsAFlatArray)
{
    std::shared_ptr<const gko::stop::CriterionFactory> iteration =
        gko::stop::Iteration::build().with_max_iters(7u).on(exec);
    std::shared_ptr<const gko::stop::CriterionFactory> residual =
        gko::stop::ResidualNorm<double>::build().with_reduction_factor(1e-5).on(
            exec);
    std::shared_ptr<const gko::stop::CriterionFactory> time =
        gko::stop::Time::build().on(exec);
    auto nested = gko::stop::combine(criterion_vector{
        iteration, gko::stop::combine(criterion_vector{residual, time})});
    auto single =
        gko::solver::Cg<double>::build().with_criteria(iteration).on(exec);
    auto combined =
        gko::solver::Cg<double>::build().with_criteria(nested).on(exec);

    auto written = write_config(*nested);

    ASSERT_EQ(written, (pnode{pnode::array_type{write_config(*iteration),
                                                write_config(*residual),
                                                write_config(*time)}}));
    ASSERT_EQ(write_config(*combined).get("criteria"), written);
    ASSERT_EQ(write_config(*single).get("criteria"),
              pnode{pnode::array_type{write_config(*iteration)}});
}


TEST_F(WriteConfigExamples, WritesNullFactoryAsEmptyNode)
{
    auto solver = gko::solver::Cg<double>::build().on(exec);

    auto written = write_config(*solver);

    ASSERT_EQ(written.get_map().count("preconditioner"), 1);
    ASSERT_EQ(written.get("preconditioner"), pnode{});
    ASSERT_EQ(written.get("criteria"), pnode{pnode::array_type{}});
}


TEST_F(WriteConfigExamples, WritesObjectParametersAsPlaceholder)
{
    std::shared_ptr<gko::LinOp> linop =
        gko::matrix::Dense<double>::create(exec);
    auto solver = gko::solver::Cg<double>::build()
                      .with_generated_preconditioner(linop)
                      .with_nullspace(linop)
                      .with_left_nullspace(linop)
                      .on(exec);

    auto written = write_config(*solver);

    ASSERT_EQ(written.get("generated_preconditioner"), pnode{"<object>"});
    ASSERT_EQ(written.get("nullspace"), pnode{"<object>"});
    ASSERT_EQ(written.get("left_nullspace"), pnode{"<object>"});
    // parse reads the placeholder as an object in the registry
    registry reg;
    reg.emplace("<object>", linop);
    auto parsed = parse(written, reg).on(exec);
    ASSERT_EQ(write_config(*parsed), written);
    ASSERT_EQ(gko::as<gko::solver::Cg<double>::Factory>(parsed.get())
                  ->get_parameters()
                  .generated_preconditioner,
              linop);
}


TEST(WriteConfig, ThrowsForTypesWithoutWriter)
{
    auto exec = gko::ReferenceExecutor::create();

    EXPECT_THROW(write_config(*gko::solver::Idr<double>::build().on(exec)),
                 gko::NotSupported);
}


TEST(WriteConfig, ThrowsForNestedTypesWithoutWriter)
{
    auto exec = gko::ReferenceExecutor::create();
    auto solver = gko::solver::Cg<double>::build()
                      .with_preconditioner(gko::solver::Idr<double>::build())
                      .on(exec);

    EXPECT_THROW(write_config(*solver), gko::NotSupported);
}


TEST(WriteConfig, ThrowsForJacobiParametersParseCanNotRead)
{
    auto exec = gko::ReferenceExecutor::create();
    using jacobi = gko::preconditioner::Jacobi<double, gko::int32>;
    gko::array<gko::int32> block_pointers{exec, {0, 2, 4}};
    gko::array<gko::precision_reduction> block_wise{
        exec, {gko::precision_reduction(0, 1), gko::precision_reduction(0, 0)}};

    EXPECT_THROW(
        write_config(
            *jacobi::build().with_block_pointers(block_pointers).on(exec)),
        gko::NotSupported);
    EXPECT_THROW(
        write_config(
            *jacobi::build().with_storage_optimization(block_wise).on(exec)),
        gko::NotSupported);
}


TEST(WriteValue, IsReadBackByGetValue)
{
    ASSERT_EQ(get_value<bool>(write_value(true)), true);
    ASSERT_EQ(get_value<gko::uint32>(write_value(gko::uint32{16})), 16u);
    ASSERT_EQ(get_value<gko::size_type>(write_value(gko::size_type{7})), 7u);
    ASSERT_EQ(get_value<double>(write_value(0.1)), 0.1);
    ASSERT_EQ(get_value<float>(write_value(0.1f)), 0.1f);
    ASSERT_EQ(static_cast<float>(
                  get_value<gko::float16>(write_value(gko::float16{0.1f}))),
              static_cast<float>(gko::float16{0.1f}));
    ASSERT_EQ(get_value<std::complex<double>>(
                  write_value(std::complex<double>{1.5, -2.0})),
              (std::complex<double>{1.5, -2.0}));
    ASSERT_EQ(get_value<std::complex<float>>(
                  write_value(std::complex<float>{0.1f, 0.2f})),
              (std::complex<float>{0.1f, 0.2f}));
    ASSERT_EQ(get_value<gko::precision_reduction>(
                  write_value(gko::precision_reduction(1, 2))),
              gko::precision_reduction(1, 2));
    for (auto mode : {gko::solver::initial_guess_mode::zero,
                      gko::solver::initial_guess_mode::rhs,
                      gko::solver::initial_guess_mode::provided}) {
        ASSERT_EQ(get_value<gko::solver::initial_guess_mode>(write_value(mode)),
                  mode);
    }
}


TEST(WriteValue, ThrowsForIntegersAboveInt64Range)
{
    ASSERT_EQ(write_value(gko::uint64{std::numeric_limits<gko::int64>::max()}),
              pnode{std::numeric_limits<gko::int64>::max()});
    ASSERT_THROW(write_value(std::numeric_limits<gko::size_type>::max()),
                 gko::NotSupported);
}


TEST(WriteValue, WritesStopModeAsBaselineString)
{
    ASSERT_EQ(write_value(gko::stop::mode::absolute), pnode{"absolute"});
    ASSERT_EQ(write_value(gko::stop::mode::initial_resnorm),
              pnode{"initial_resnorm"});
    ASSERT_EQ(write_value(gko::stop::mode::rhs_norm), pnode{"rhs_norm"});
}


TEST(RecordRequestedKeys, RecordsTheKeysParseRequests)
{
    record_requested_keys recorder;

    parse(pnode{{{"type", pnode{"preconditioner::Jacobi"}}}}, registry{});

    ASSERT_EQ(recorder.keys(),
              (std::set<std::string>{"accuracy", "aggregate_l1",
                                     "max_block_size", "max_block_stride",
                                     "skip_sorting", "storage_optimization"}));
}


TEST(RecordRequestedKeys, RestoresThePreviousRecorder)
{
    ASSERT_EQ(detail::requested_keys(), nullptr);
    {
        record_requested_keys outer;
        {
            record_requested_keys inner;
            ASSERT_EQ(detail::requested_keys(), &inner.keys());

            parse(pnode{{{"type", pnode{"preconditioner::Jacobi"}}}},
                  registry{});

            ASSERT_FALSE(inner.keys().empty());
        }
        ASSERT_EQ(detail::requested_keys(), &outer.keys());
        ASSERT_TRUE(outer.keys().empty());
    }
    ASSERT_EQ(detail::requested_keys(), nullptr);
}


}  // namespace
