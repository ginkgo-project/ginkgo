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
#include <ginkgo/core/factorization/ic.hpp>
#include <ginkgo/core/factorization/ilu.hpp>
#include <ginkgo/core/factorization/par_ic.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/preconditioner/ic.hpp>
#include <ginkgo/core/preconditioner/ilu.hpp>
#include <ginkgo/core/preconditioner/isai.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/idr.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/triangular.hpp>
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


// The keys that parse reads but write_config does not write: the class and
// template selection, which write_config writes separately or not at all, and
// the object-valued parameters, which it omits when they are null.
const std::set<std::string> not_written_by_default{"type",
                                                   "value_type",
                                                   "index_type",
                                                   "generated_preconditioner",
                                                   "nullspace",
                                                   "left_nullspace",
                                                   "generated_solver"};


// An empty node would be null in JSON or YAML, which their readers reject.
bool contains_empty_node(const pnode& node)
{
    switch (node.get_tag()) {
    case pnode::tag_t::empty:
        return true;
    case pnode::tag_t::array:
        for (const auto& item : node.get_array()) {
            if (contains_empty_node(item)) {
                return true;
            }
        }
        return false;
    case pnode::tag_t::map:
        for (const auto& item : node.get_map()) {
            if (contains_empty_node(item.second)) {
                return true;
            }
        }
        return false;
    default:
        return false;
    }
}


/**
 * Describes one LinOp type for the typed tests.
 *
 * - `type`: the registry key that selects the type in parse.
 * - `index_type`: the index type parse has to be given through the
 *   type_descriptor; parse does not accept it in the configuration.
 * - `non_default`: a factory with a non-default value for every parameter that
 *   parse can read back from a configuration.
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


template <typename ValueType>
struct write_case<gko::solver::Bicgstab<ValueType>> {
    static constexpr const char* type = "solver::Bicgstab";
    using index_type = gko::int32;

    static std::unique_ptr<gko::LinOpFactory> non_default(
        std::shared_ptr<const gko::Executor> exec)
    {
        return gko::solver::Bicgstab<ValueType>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(7u))
            .with_preconditioner(
                gko::preconditioner::Jacobi<ValueType, index_type>::build()
                    .with_max_block_size(16u))
            .on(exec);
    }
};


template <typename ValueType>
struct write_case<gko::solver::Gmres<ValueType>> {
    static constexpr const char* type = "solver::Gmres";
    using index_type = gko::int32;

    static std::unique_ptr<gko::LinOpFactory> non_default(
        std::shared_ptr<const gko::Executor> exec)
    {
        return gko::solver::Gmres<ValueType>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(7u))
            .with_preconditioner(
                gko::preconditioner::Jacobi<ValueType, index_type>::build()
                    .with_max_block_size(16u))
            .with_krylov_dim(20u)
            .with_restart_ratio(gko::remove_complex<ValueType>{0.5})
            .with_flexible(true)
            .with_ortho_method(gko::solver::gmres::ortho_method::cgs2)
            .on(exec);
    }
};


template <typename ValueType>
struct write_case<gko::solver::Ir<ValueType>> {
    static constexpr const char* type = "solver::Ir";
    using index_type = gko::int32;

    static std::unique_ptr<gko::LinOpFactory> non_default(
        std::shared_ptr<const gko::Executor> exec)
    {
        return gko::solver::Ir<ValueType>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(7u))
            .with_solver(
                gko::preconditioner::Jacobi<ValueType, index_type>::build()
                    .with_max_block_size(1u))
            .with_relaxation_factor(ValueType{0.5})
            .with_default_initial_guess(gko::solver::initial_guess_mode::zero)
            .with_inner_initial_guess(gko::solver::initial_guess_mode::rhs)
            .on(exec);
    }
};


template <typename ValueType, typename IndexType>
struct write_case<gko::solver::LowerTrs<ValueType, IndexType>> {
    static constexpr const char* type = "solver::LowerTrs";
    using index_type = IndexType;

    static std::unique_ptr<gko::LinOpFactory> non_default(
        std::shared_ptr<const gko::Executor> exec)
    {
        return gko::solver::LowerTrs<ValueType, IndexType>::build()
            .with_num_rhs(3u)
            .with_unit_diagonal(true)
            .with_algorithm(gko::solver::trisolve_algorithm::syncfree)
            .on(exec);
    }
};


template <typename ValueType, typename IndexType>
struct write_case<gko::solver::UpperTrs<ValueType, IndexType>> {
    static constexpr const char* type = "solver::UpperTrs";
    using index_type = IndexType;

    static std::unique_ptr<gko::LinOpFactory> non_default(
        std::shared_ptr<const gko::Executor> exec)
    {
        return gko::solver::UpperTrs<ValueType, IndexType>::build()
            .with_num_rhs(3u)
            .with_unit_diagonal(true)
            .with_algorithm(gko::solver::trisolve_algorithm::syncfree)
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


template <typename ValueType, typename IndexType>
struct write_case<gko::factorization::Ilu<ValueType, IndexType>> {
    static constexpr const char* type = "factorization::Ilu";
    using index_type = IndexType;

    static std::unique_ptr<gko::LinOpFactory> non_default(
        std::shared_ptr<const gko::Executor> exec)
    {
        return gko::factorization::Ilu<ValueType, IndexType>::build()
            .with_l_strategy(gko::matrix::csr::spmv_strategy::merge_path)
            .with_u_strategy(gko::matrix::csr::spmv_strategy::load_balance)
            .with_skip_sorting(true)
            .with_algorithm(gko::factorization::incomplete_algorithm::syncfree)
            .on(exec);
    }
};


template <typename ValueType, typename IndexType>
struct write_case<gko::factorization::Ic<ValueType, IndexType>> {
    static constexpr const char* type = "factorization::Ic";
    using index_type = IndexType;

    static std::unique_ptr<gko::LinOpFactory> non_default(
        std::shared_ptr<const gko::Executor> exec)
    {
        return gko::factorization::Ic<ValueType, IndexType>::build()
            .with_l_strategy(gko::matrix::csr::spmv_strategy::merge_path)
            .with_skip_sorting(true)
            .with_both_factors(false)
            .with_algorithm(gko::factorization::incomplete_algorithm::syncfree)
            .on(exec);
    }
};


template <typename ValueType, bool ReverseApply, typename IndexType>
struct write_case<
    gko::preconditioner::Ilu<ValueType, ReverseApply, IndexType>> {
    static constexpr const char* type = "preconditioner::Ilu";
    using index_type = IndexType;

    static std::unique_ptr<gko::LinOpFactory> non_default(
        std::shared_ptr<const gko::Executor> exec)
    {
        return gko::preconditioner::Ilu<ValueType, ReverseApply,
                                        IndexType>::build()
            .with_factorization(
                gko::factorization::Ilu<ValueType, IndexType>::build()
                    .with_skip_sorting(true))
            .with_l_solver(
                gko::solver::LowerTrs<ValueType, IndexType>::build()
                    .with_algorithm(gko::solver::trisolve_algorithm::syncfree))
            .with_u_solver(
                gko::solver::UpperTrs<ValueType, IndexType>::build()
                    .with_algorithm(gko::solver::trisolve_algorithm::syncfree))
            .on(exec);
    }
};


template <typename ValueType, typename IndexType>
struct write_case<gko::preconditioner::Ic<ValueType, IndexType>> {
    static constexpr const char* type = "preconditioner::Ic";
    using index_type = IndexType;

    static std::unique_ptr<gko::LinOpFactory> non_default(
        std::shared_ptr<const gko::Executor> exec)
    {
        return gko::preconditioner::Ic<ValueType, IndexType>::build()
            .with_factorization(
                gko::factorization::Ic<ValueType, IndexType>::build()
                    .with_skip_sorting(true))
            .with_l_solver(
                gko::solver::LowerTrs<ValueType, IndexType>::build()
                    .with_algorithm(gko::solver::trisolve_algorithm::syncfree))
            .on(exec);
    }
};


template <gko::preconditioner::isai_type IsaiType, typename ValueType,
          typename IndexType>
struct write_case<gko::preconditioner::Isai<IsaiType, ValueType, IndexType>> {
    static constexpr const char* type = "preconditioner::Isai";
    using index_type = IndexType;

    static std::unique_ptr<gko::LinOpFactory> non_default(
        std::shared_ptr<const gko::Executor> exec)
    {
        return gko::preconditioner::Isai<IsaiType, ValueType,
                                         IndexType>::build()
            .with_skip_sorting(true)
            .with_sparsity_power(2)
            .with_excess_limit(32u)
            .with_excess_solver_factory(
                gko::solver::Gmres<ValueType>::build().with_krylov_dim(5u))
            .with_excess_solver_reduction(gko::remove_complex<ValueType>{1e-4})
            .on(exec);
    }
};


/**
 * The keys besides "type" with which parse selects the instantiation of `T`,
 * e.g. preconditioner::Ilu's "reverse_apply". parse reads them without
 * config_check_decorator, so record_requested_keys does not see them.
 */
template <typename T>
struct selection_keys {
    static pnode::map_type get() { return {}; }
};


template <typename ValueType, bool ReverseApply, typename IndexType>
struct selection_keys<
    gko::preconditioner::Ilu<ValueType, ReverseApply, IndexType>> {
    static pnode::map_type get()
    {
        return {{"reverse_apply", pnode{ReverseApply}}};
    }
};


template <gko::preconditioner::isai_type IsaiType, typename ValueType,
          typename IndexType>
struct selection_keys<
    gko::preconditioner::Isai<IsaiType, ValueType, IndexType>> {
    static pnode::map_type get()
    {
        using gko::preconditioner::isai_type;
        const char* name = IsaiType == isai_type::lower     ? "lower"
                           : IsaiType == isai_type::upper   ? "upper"
                           : IsaiType == isai_type::general ? "general"
                                                            : "spd";
        return {{"isai_type", pnode{name}}};
    }
};


// The configuration that selects T in parse and sets none of its parameters.
template <typename T>
pnode selecting_config()
{
    auto map = selection_keys<T>::get();
    map["type"] = pnode{write_case<T>::type};
    return pnode{map};
}


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

using WriteConfigTypes = ::testing::Types<
    gko::solver::Cg<double>, gko::solver::Cg<float>,
    gko::solver::Cg<std::complex<float>>, gko::solver::Bicgstab<double>,
    gko::solver::Bicgstab<std::complex<float>>, gko::solver::Gmres<double>,
    gko::solver::Gmres<float>, gko::solver::Gmres<std::complex<float>>,
    gko::solver::Ir<double>, gko::solver::Ir<float>,
    gko::solver::Ir<std::complex<float>>,
    gko::solver::LowerTrs<double, gko::int32>,
    gko::solver::LowerTrs<float, gko::int64>,
    gko::solver::UpperTrs<double, gko::int32>,
    gko::solver::UpperTrs<float, gko::int64>,
    gko::preconditioner::Jacobi<double, gko::int32>,
    gko::preconditioner::Jacobi<float, gko::int64>,
    gko::factorization::Ilu<double, gko::int32>,
    gko::factorization::Ilu<float, gko::int64>,
    gko::factorization::Ic<double, gko::int32>,
    gko::factorization::Ic<std::complex<float>, gko::int64>,
    gko::preconditioner::Ilu<double, false, gko::int32>,
    gko::preconditioner::Ilu<float, true, gko::int64>,
    gko::preconditioner::Ic<double, gko::int32>,
    gko::preconditioner::Ic<std::complex<float>, gko::int64>,
    gko::preconditioner::LowerIsai<double, gko::int32>,
    gko::preconditioner::UpperIsai<float, gko::int64>,
    gko::preconditioner::GeneralIsai<double, gko::int32>,
    gko::preconditioner::SpdIsai<std::complex<float>, gko::int32>>;

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


TYPED_TEST(WriteConfig, WritesTheKeysThatSelectTheInstantiation)
{
    using linop_type = typename TestFixture::linop_type;
    using write_case = typename TestFixture::write_case;

    const auto selection = selecting_config<linop_type>();

    auto written_default = write_config(*linop_type::build().on(this->exec));
    auto written = write_config(*write_case::non_default(this->exec));

    for (const auto& [key, value] : selection.get_map()) {
        EXPECT_EQ(written_default.get(key), value)
            << "write_config does not write the key " << key;
        EXPECT_EQ(written.get(key), value)
            << "write_config does not write the key " << key;
    }
}


TYPED_TEST(WriteConfig, WritesEveryParsedKey)
{
    using linop_type = typename TestFixture::linop_type;
    using write_case = typename TestFixture::write_case;
    std::set<std::string> parsed_keys;
    {
        record_requested_keys recorder;
        parse(selecting_config<linop_type>(), registry{}, this->td());
        parsed_keys = recorder.keys();
    }

    // non_default sets every parameter, the default leaves the nested
    // factories unset
    auto written = write_config(*write_case::non_default(this->exec));
    auto written_default = write_config(*linop_type::build().on(this->exec));

    ASSERT_FALSE(parsed_keys.empty());
    for (const auto& key : parsed_keys) {
        if (not_written_by_default.count(key) == 0) {
            EXPECT_EQ(written.get_map().count(key), 1)
                << "write_config does not write the key " << key;
            // only an unset nested factory may be left out of the default
            if (written_default.get_map().count(key) == 0) {
                EXPECT_EQ(written.get(key).get_tag(), pnode::tag_t::map)
                    << "write_config leaves out the default of the key " << key;
            }
        }
    }
}


TYPED_TEST(WriteConfig, WritesNoEmptyNode)
{
    using linop_type = typename TestFixture::linop_type;
    using write_case = typename TestFixture::write_case;

    auto written_default = write_config(*linop_type::build().on(this->exec));
    auto written = write_config(*write_case::non_default(this->exec));

    ASSERT_FALSE(contains_empty_node(written_default))
        << ::testing::PrintToString(written_default);
    ASSERT_FALSE(contains_empty_node(written))
        << ::testing::PrintToString(written);
}


TYPED_TEST(WriteConfig, WritesTheValueOfEveryParsedKey)
{
    using linop_type = typename TestFixture::linop_type;
    using write_case = typename TestFixture::write_case;
    std::set<std::string> parsed_keys;
    {
        record_requested_keys recorder;
        parse(selecting_config<linop_type>(), registry{}, this->td());
        parsed_keys = recorder.keys();
    }

    auto written_default = write_config(*linop_type::build().on(this->exec));
    auto written = write_config(*write_case::non_default(this->exec));

    // non_default sets every parameter to a non-default value, so every key
    // differs, unless the writer does not write the parameter's value
    for (const auto& key : parsed_keys) {
        if (not_written_by_default.count(key) == 0) {
            EXPECT_NE(written.get(key), written_default.get(key))
                << "write_config does not write the value of the key " << key;
        }
    }
}


/**
 * Describes one stopping criterion for the typed criterion tests, like
 * write_case does for LinOp types.
 */
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


/**
 * Copies of the factories of the same names in
 * benchmark/utils/preconditioners.hpp, with the benchmark's default types
 * (etype double, itype int32) and flags (-isai_power 1). The core tests do not
 * include the benchmark headers.
 */
namespace benchmark_precond {


using etype = double;
using itype = gko::int32;
constexpr int isai_power = 1;


std::unique_ptr<gko::LinOpFactory> ic(std::shared_ptr<const gko::Executor> exec)
{
    auto fact =
        gko::share(gko::factorization::Ic<etype, itype>::build().on(exec));
    return gko::preconditioner::Ic<etype, itype>::build()
        .with_factorization(fact)
        .on(exec);
}


std::unique_ptr<gko::LinOpFactory> ilu(
    std::shared_ptr<const gko::Executor> exec)
{
    auto fact =
        gko::share(gko::factorization::Ilu<etype, itype>::build().on(exec));
    return gko::preconditioner::Ilu<etype, false, itype>::build()
        .with_factorization(fact)
        .on(exec);
}


std::unique_ptr<gko::LinOpFactory> ic_isai(
    std::shared_ptr<const gko::Executor> exec)
{
    auto fact =
        gko::share(gko::factorization::Ic<etype, itype>::build().on(exec));
    auto lisai =
        gko::share(gko::preconditioner::LowerIsai<etype, itype>::build()
                       .with_sparsity_power(isai_power)
                       .on(exec));
    return gko::preconditioner::Ic<etype, itype>::build()
        .with_factorization(fact)
        .with_l_solver(lisai)
        .on(exec);
}


std::unique_ptr<gko::LinOpFactory> ilu_isai(
    std::shared_ptr<const gko::Executor> exec)
{
    auto fact =
        gko::share(gko::factorization::Ilu<etype, itype>::build().on(exec));
    auto lisai =
        gko::share(gko::preconditioner::LowerIsai<etype, itype>::build()
                       .with_sparsity_power(isai_power)
                       .on(exec));
    auto uisai =
        gko::share(gko::preconditioner::UpperIsai<etype, itype>::build()
                       .with_sparsity_power(isai_power)
                       .on(exec));
    return gko::preconditioner::Ilu<etype, false, itype>::build()
        .with_factorization(fact)
        .with_l_solver(lisai)
        .with_u_solver(uisai)
        .on(exec);
}


std::unique_ptr<gko::LinOpFactory> ilu_syncfree(
    std::shared_ptr<const gko::Executor> exec)
{
    auto fact = gko::share(
        gko::factorization::Ilu<etype, itype>::build()
            .with_algorithm(gko::factorization::incomplete_algorithm::syncfree)
            .on(exec));
    return gko::preconditioner::Ilu<etype, false, itype>::build()
        .with_factorization(fact)
        .with_l_solver(
            gko::solver::LowerTrs<etype, itype>::build().with_algorithm(
                gko::solver::trisolve_algorithm::syncfree))
        .with_u_solver(
            gko::solver::UpperTrs<etype, itype>::build().with_algorithm(
                gko::solver::trisolve_algorithm::syncfree))
        .on(exec);
}


std::unique_ptr<gko::LinOpFactory> ic_syncfree(
    std::shared_ptr<const gko::Executor> exec)
{
    auto fact = gko::share(
        gko::factorization::Ic<etype, itype>::build()
            .with_algorithm(gko::factorization::incomplete_algorithm::syncfree)
            .on(exec));
    return gko::preconditioner::Ic<etype, itype>::build()
        .with_factorization(fact)
        .with_l_solver(
            gko::solver::LowerTrs<etype, itype>::build().with_algorithm(
                gko::solver::trisolve_algorithm::syncfree))
        .on(exec);
}


std::unique_ptr<gko::LinOpFactory> general_isai(
    std::shared_ptr<const gko::Executor> exec)
{
    return gko::preconditioner::GeneralIsai<etype, itype>::build()
        .with_sparsity_power(isai_power)
        .on(exec);
}


std::unique_ptr<gko::LinOpFactory> spd_isai(
    std::shared_ptr<const gko::Executor> exec)
{
    return gko::preconditioner::SpdIsai<etype, itype>::build()
        .with_sparsity_power(isai_power)
        .on(exec);
}


// The written trees of the parts of these factories
pnode::map_type factorization_ilu(const char* algorithm)
{
    return {{"type", pnode{"factorization::Ilu"}},
            {"value_type", pnode{"float64"}},
            {"l_strategy", pnode{"classical"}},
            {"u_strategy", pnode{"classical"}},
            {"skip_sorting", pnode{false}},
            {"algorithm", pnode{algorithm}}};
}


pnode::map_type factorization_ic(const char* algorithm)
{
    return {
        {"type", pnode{"factorization::Ic"}}, {"value_type", pnode{"float64"}},
        {"l_strategy", pnode{"classical"}},   {"skip_sorting", pnode{false}},
        {"both_factors", pnode{true}},        {"algorithm", pnode{algorithm}}};
}


pnode::map_type syncfree_trs(const char* type)
{
    return {{"type", pnode{type}},
            {"value_type", pnode{"float64"}},
            {"num_rhs", pnode{1}},
            {"unit_diagonal", pnode{false}},
            {"algorithm", pnode{"syncfree"}}};
}


pnode::map_type isai(const char* isai_type)
{
    return {{"type", pnode{"preconditioner::Isai"}},
            {"value_type", pnode{"float64"}},
            {"isai_type", pnode{isai_type}},
            {"skip_sorting", pnode{false}},
            {"sparsity_power", pnode{1}},
            {"excess_limit", pnode{0}},
            {"excess_solver_reduction", pnode{1e-6}}};
}


}  // namespace benchmark_precond


class WriteConfigExamples : public ::testing::Test {
protected:
    WriteConfigExamples() : exec(gko::ReferenceExecutor::create()) {}

    // write, parse (with the default index type int32) and write again
    template <typename FactoryType>
    void assert_round_trip(const gko::LinOpFactory& factory)
    {
        auto written = write_config(factory);

        auto parsed = parse(written, registry{}).on(exec);

        ASSERT_NE(dynamic_cast<const FactoryType*>(parsed.get()), nullptr);
        ASSERT_EQ(write_config(*parsed), written);
    }

    std::shared_ptr<const gko::Executor> exec;
};


TEST_F(WriteConfigExamples, WritesBenchmarkIluSyncfree)
{
    using namespace benchmark_precond;
    auto factory = ilu_syncfree(exec);

    auto written = write_config(*factory);

    pnode expected{{{"type", pnode{"preconditioner::Ilu"}},
                    {"value_type", pnode{"float64"}},
                    {"reverse_apply", pnode{false}},
                    {"factorization", pnode{factorization_ilu("syncfree")}},
                    {"l_solver", pnode{syncfree_trs("solver::LowerTrs")}},
                    {"u_solver", pnode{syncfree_trs("solver::UpperTrs")}}}};
    ASSERT_EQ(written, expected);
    this->assert_round_trip<
        gko::preconditioner::Ilu<etype, false, itype>::Factory>(*factory);
}


TEST_F(WriteConfigExamples, WritesBenchmarkIcSyncfree)
{
    using namespace benchmark_precond;
    auto factory = ic_syncfree(exec);

    auto written = write_config(*factory);

    pnode expected{{{"type", pnode{"preconditioner::Ic"}},
                    {"value_type", pnode{"float64"}},
                    {"factorization", pnode{factorization_ic("syncfree")}},
                    {"l_solver", pnode{syncfree_trs("solver::LowerTrs")}}}};
    ASSERT_EQ(written, expected);
    this->assert_round_trip<gko::preconditioner::Ic<etype, itype>::Factory>(
        *factory);
}


TEST_F(WriteConfigExamples, WritesBenchmarkIlu)
{
    using namespace benchmark_precond;
    auto factory = ilu(exec);

    auto written = write_config(*factory);

    // the unset solvers are the default triangular solvers
    pnode expected{{{"type", pnode{"preconditioner::Ilu"}},
                    {"value_type", pnode{"float64"}},
                    {"reverse_apply", pnode{false}},
                    {"factorization", pnode{factorization_ilu("sparselib")}}}};
    ASSERT_EQ(written, expected);
    this->assert_round_trip<
        gko::preconditioner::Ilu<etype, false, itype>::Factory>(*factory);
}


TEST_F(WriteConfigExamples, WritesBenchmarkIc)
{
    using namespace benchmark_precond;
    auto factory = ic(exec);

    auto written = write_config(*factory);

    pnode expected{{{"type", pnode{"preconditioner::Ic"}},
                    {"value_type", pnode{"float64"}},
                    {"factorization", pnode{factorization_ic("sparselib")}}}};
    ASSERT_EQ(written, expected);
    this->assert_round_trip<gko::preconditioner::Ic<etype, itype>::Factory>(
        *factory);
}


TEST_F(WriteConfigExamples, WritesBenchmarkGeneralIsai)
{
    using namespace benchmark_precond;
    auto factory = general_isai(exec);

    auto written = write_config(*factory);

    // the unset excess solver is the default one
    ASSERT_EQ(written, pnode{isai("general")});
    this->assert_round_trip<
        gko::preconditioner::GeneralIsai<etype, itype>::Factory>(*factory);
}


TEST_F(WriteConfigExamples, WritesBenchmarkSpdIsai)
{
    using namespace benchmark_precond;
    auto factory = spd_isai(exec);

    auto written = write_config(*factory);

    ASSERT_EQ(written, pnode{isai("spd")});
    this->assert_round_trip<
        gko::preconditioner::SpdIsai<etype, itype>::Factory>(*factory);
}


TEST_F(WriteConfigExamples, WritesBenchmarkIluIsaiAndIcIsai)
{
    using namespace benchmark_precond;
    auto ilu_factory = ilu_isai(exec);
    auto ic_factory = ic_isai(exec);

    auto written_ilu = write_config(*ilu_factory);
    auto written_ic = write_config(*ic_factory);

    ASSERT_EQ(written_ilu.get("l_solver"), pnode{isai("lower")});
    ASSERT_EQ(written_ilu.get("u_solver"), pnode{isai("upper")});
    ASSERT_EQ(written_ilu.get("factorization"),
              pnode{factorization_ilu("sparselib")});
    ASSERT_EQ(written_ic.get("l_solver"), pnode{isai("lower")});
    ASSERT_EQ(written_ic.get("factorization"),
              pnode{factorization_ic("sparselib")});
    this->assert_round_trip<
        gko::preconditioner::Ilu<etype, false, itype>::Factory>(*ilu_factory);
    this->assert_round_trip<gko::preconditioner::Ic<etype, itype>::Factory>(
        *ic_factory);
}


TEST_F(WriteConfigExamples, WritesIluWithReverseApply)
{
    auto factory =
        gko::preconditioner::Ilu<double, true, gko::int32>::build().on(exec);

    auto written = write_config(*factory);

    ASSERT_EQ(written, (pnode{{{"type", pnode{"preconditioner::Ilu"}},
                               {"value_type", pnode{"float64"}},
                               {"reverse_apply", pnode{true}}}}));
    this->assert_round_trip<
        gko::preconditioner::Ilu<double, true, gko::int32>::Factory>(*factory);
}


TEST_F(WriteConfigExamples, WritesFactorizationEnumsAsStrings)
{
    using gko::factorization::incomplete_algorithm;
    using gko::matrix::csr::spmv_strategy;
    using ilu_type = gko::factorization::Ilu<double, gko::int32>;
    using ic_type = gko::factorization::Ic<double, gko::int32>;
    for (auto [strategy, name] :
         {std::make_pair(spmv_strategy::automatic, "automatic"),
          std::make_pair(spmv_strategy::load_balance, "load_balance"),
          std::make_pair(spmv_strategy::merge_path, "merge_path"),
          std::make_pair(spmv_strategy::classical, "classical"),
          std::make_pair(spmv_strategy::sparselib, "sparselib")}) {
        auto ilu = ilu_type::build()
                       .with_l_strategy(strategy)
                       .with_u_strategy(strategy)
                       .on(exec);
        auto ic = ic_type::build().with_l_strategy(strategy).on(exec);

        auto written_ilu = write_config(*ilu);
        auto written_ic = write_config(*ic);

        ASSERT_EQ(written_ilu.get("l_strategy"), pnode{name});
        ASSERT_EQ(written_ilu.get("u_strategy"), pnode{name});
        ASSERT_EQ(written_ic.get("l_strategy"), pnode{name});
        auto parsed_ilu = parse(written_ilu, registry{}).on(exec);
        auto parsed_ic = parse(written_ic, registry{}).on(exec);
        ASSERT_EQ(gko::as<ilu_type::Factory>(parsed_ilu.get())
                      ->get_parameters()
                      .l_strategy,
                  strategy);
        ASSERT_EQ(gko::as<ilu_type::Factory>(parsed_ilu.get())
                      ->get_parameters()
                      .u_strategy,
                  strategy);
        ASSERT_EQ(gko::as<ic_type::Factory>(parsed_ic.get())
                      ->get_parameters()
                      .l_strategy,
                  strategy);
    }
    for (auto [algorithm, name] :
         {std::make_pair(incomplete_algorithm::sparselib, "sparselib"),
          std::make_pair(incomplete_algorithm::syncfree, "syncfree")}) {
        auto ilu = ilu_type::build().with_algorithm(algorithm).on(exec);
        auto ic = ic_type::build().with_algorithm(algorithm).on(exec);

        ASSERT_EQ(write_config(*ilu).get("algorithm"), pnode{name});
        ASSERT_EQ(write_config(*ic).get("algorithm"), pnode{name});
    }
}


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


TEST_F(WriteConfigExamples, WritesCombinedCriteriaAsFlatArray)
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
    auto solver =
        gko::solver::Cg<double>::build().with_criteria(nested).on(exec);

    auto written = write_config(*nested);

    ASSERT_EQ(written, (pnode{pnode::array_type{write_config(*iteration),
                                                write_config(*residual),
                                                write_config(*time)}}));
    ASSERT_EQ(write_config(*solver).get("criteria"), written);
}


TEST_F(WriteConfigExamples, WritesCriteriaAsArrayForASingleCriterion)
{
    std::shared_ptr<const gko::stop::CriterionFactory> iteration =
        gko::stop::Iteration::build().with_max_iters(7u).on(exec);
    auto solver =
        gko::solver::Cg<double>::build().with_criteria(iteration).on(exec);

    auto written = write_config(*solver);

    ASSERT_EQ(written.get("criteria"),
              pnode{pnode::array_type{write_config(*iteration)}});
}


TEST_F(WriteConfigExamples, WritesIrWithScalarJacobiSolver)
{
    auto factory =
        gko::solver::Ir<double>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(7u))
            .with_solver(
                gko::preconditioner::Jacobi<double, gko::int32>::build()
                    .with_max_block_size(1u))
            .on(exec);

    auto written = write_config(*factory);

    ASSERT_EQ(written.get("type"), pnode{"solver::Ir"});
    ASSERT_EQ(written.get("solver").get("type"),
              pnode{"preconditioner::Jacobi"});
    ASSERT_EQ(written.get("solver").get("max_block_size"), pnode{1});
    ASSERT_EQ(written.get("relaxation_factor"), pnode{1.0});
    ASSERT_EQ(written.get("default_initial_guess"), pnode{"provided"});
    ASSERT_EQ(written.get("inner_initial_guess"), pnode{"zero"});
}


TEST_F(WriteConfigExamples, OmitsIdentityIrSolver)
{
    // an unset Ir solver is the identity
    auto unset = gko::solver::Ir<double>::build().on(exec);
    auto identity =
        gko::solver::Ir<double>::build()
            .with_solver(gko::matrix::IdentityFactory<double>::create(exec))
            .on(exec);

    ASSERT_EQ(write_config(*unset).get_map().count("solver"), 0);
    ASSERT_EQ(write_config(*identity), write_config(*unset));
}


TEST_F(WriteConfigExamples, WritesIrGeneratedSolverAsPlaceholder)
{
    std::shared_ptr<gko::LinOp> linop =
        gko::matrix::Dense<double>::create(exec);
    auto factory =
        gko::solver::Ir<double>::build().with_generated_solver(linop).on(exec);

    auto written = write_config(*factory);

    ASSERT_EQ(written.get("generated_solver"), pnode{"<object>"});
    registry reg;
    reg.emplace("<object>", linop);
    auto parsed = parse(written, reg).on(exec);
    ASSERT_EQ(write_config(*parsed), written);
    ASSERT_EQ(gko::as<gko::solver::Ir<double>::Factory>(parsed.get())
                  ->get_parameters()
                  .generated_solver,
              linop);
    ASSERT_EQ(write_config(*gko::solver::Ir<double>::build().on(exec))
                  .get_map()
                  .count("generated_solver"),
              0);
}


TEST_F(WriteConfigExamples, WritesGmresAndTriangularEnumsAsStrings)
{
    using gko::solver::trisolve_algorithm;
    using gko::solver::gmres::ortho_method;
    for (auto [method, name] : {std::make_pair(ortho_method::mgs, "mgs"),
                                std::make_pair(ortho_method::cgs, "cgs"),
                                std::make_pair(ortho_method::cgs2, "cgs2")}) {
        auto gmres =
            gko::solver::Gmres<double>::build().with_ortho_method(method).on(
                exec);

        ASSERT_EQ(write_config(*gmres).get("ortho_method"), pnode{name});
    }
    for (auto [algorithm, name] :
         {std::make_pair(trisolve_algorithm::sparselib, "sparselib"),
          std::make_pair(trisolve_algorithm::syncfree, "syncfree")}) {
        auto lower = gko::solver::LowerTrs<double, gko::int32>::build()
                         .with_algorithm(algorithm)
                         .on(exec);
        auto upper = gko::solver::UpperTrs<double, gko::int32>::build()
                         .with_algorithm(algorithm)
                         .on(exec);

        ASSERT_EQ(write_config(*lower).get("algorithm"), pnode{name});
        ASSERT_EQ(write_config(*upper).get("algorithm"), pnode{name});
    }
}


TEST_F(WriteConfigExamples, OmitsNullFactory)
{
    auto solver = gko::solver::Cg<double>::build().on(exec);

    auto written = write_config(*solver);

    ASSERT_EQ(written.get_map().count("preconditioner"), 0);
    ASSERT_EQ(written.get("criteria"), pnode{pnode::array_type{}});
    ASSERT_FALSE(contains_empty_node(written));
}


TEST_F(WriteConfigExamples, OmitsIdentityPreconditioner)
{
    // an unset preconditioner is the identity, of the solver's value type
    for (std::shared_ptr<const gko::LinOpFactory> identity :
         {std::shared_ptr<const gko::LinOpFactory>{
              gko::matrix::IdentityFactory<double>::create(exec)},
          std::shared_ptr<const gko::LinOpFactory>{
              gko::matrix::IdentityFactory<float>::create(exec)}}) {
        auto solver =
            gko::solver::Cg<double>::build().with_preconditioner(identity).on(
                exec);

        auto written = write_config(*solver);

        ASSERT_EQ(written.get_map().count("preconditioner"), 0);
        ASSERT_EQ(written,
                  write_config(*gko::solver::Cg<double>::build().on(exec)));
        auto parsed = parse(written, registry{}).on(exec);
        ASSERT_NE(
            dynamic_cast<const gko::solver::Cg<double>::Factory*>(parsed.get()),
            nullptr);
        ASSERT_EQ(write_config(*parsed), written);
    }
}


TEST_F(WriteConfigExamples, OmitsNullObjectParameters)
{
    auto solver = gko::solver::Cg<double>::build().on(exec);

    auto written = write_config(*solver);

    ASSERT_EQ(written.get_map().count("generated_preconditioner"), 0);
    ASSERT_EQ(written.get_map().count("nullspace"), 0);
    ASSERT_EQ(written.get_map().count("left_nullspace"), 0);
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
    // parse reads the placeholder back once the registry stores an object
    // under that name
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


TEST(WriteConfig, ThrowsForIdentityFactory)
{
    auto exec = gko::ReferenceExecutor::create();

    EXPECT_THROW(
        write_config(*gko::matrix::IdentityFactory<double>::create(exec)),
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


TEST(WriteConfig, ThrowsForIcWithParIcFactorization)
{
    // the benchmark's paric: ParIc has no writer
    auto exec = gko::ReferenceExecutor::create();
    auto ic = gko::preconditioner::Ic<double, gko::int32>::build()
                  .with_factorization(
                      gko::factorization::ParIc<double, gko::int32>::build()
                          .with_iterations(5u)
                          .with_skip_sorting(true))
                  .on(exec);

    EXPECT_THROW(write_config(*ic), gko::NotSupported);
}


TEST(WriteConfig, ThrowsForIdentityIluSolver)
{
    // an unset l_solver is a LowerTrs, not the identity, so an IdentityFactory
    // can not be written as an unset l_solver
    auto exec = gko::ReferenceExecutor::create();
    auto ilu =
        gko::preconditioner::Ilu<double, false, gko::int32>::build()
            .with_l_solver(gko::matrix::IdentityFactory<double>::create(exec))
            .on(exec);

    EXPECT_THROW(write_config(*ilu), gko::NotSupported);
}


TEST(WriteConfig, ThrowsForJacobiBlockPointers)
{
    auto exec = gko::ReferenceExecutor::create();
    gko::array<gko::int32> block_pointers{exec, {0, 2, 4}};
    auto jacobi = gko::preconditioner::Jacobi<double, gko::int32>::build()
                      .with_block_pointers(block_pointers)
                      .on(exec);

    EXPECT_THROW(write_config(*jacobi), gko::NotSupported);
}


TEST(WriteConfig, ThrowsForJacobiBlockWiseStorageOptimization)
{
    auto exec = gko::ReferenceExecutor::create();
    gko::array<gko::precision_reduction> block_wise{
        exec, {gko::precision_reduction(0, 1), gko::precision_reduction(0, 0)}};
    auto jacobi = gko::preconditioner::Jacobi<double, gko::int32>::build()
                      .with_storage_optimization(block_wise)
                      .on(exec);

    EXPECT_THROW(write_config(*jacobi), gko::NotSupported);
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


TEST(WriteValue, WritesComplexAsRealImagArray)
{
    ASSERT_EQ(write_value(std::complex<double>{1.5, -2.0}),
              (pnode{pnode::array_type{pnode{1.5}, pnode{-2.0}}}));
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
