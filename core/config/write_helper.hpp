// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_CONFIG_WRITE_HELPER_HPP_
#define GKO_CORE_CONFIG_WRITE_HELPER_HPP_


#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/property_tree.hpp>
#include <ginkgo/core/config/write_config.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/criterion.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/core/stop/time.hpp>

#include "core/config/type_descriptor_helper.hpp"


namespace gko {
namespace config {


/**
 * write_value writes a parameter value in the encoding that get_value reads.
 */
inline pnode write_value(bool value) { return pnode{value}; }


/**
 * write_value for integers, which throws NotSupported above the int64 range
 */
template <typename IndexType>
inline std::enable_if_t<std::is_integral<IndexType>::value &&
                            !std::is_same<IndexType, bool>::value,
                        pnode>
write_value(IndexType value)
{
    if (std::is_unsigned<IndexType>::value &&
        static_cast<std::uintmax_t>(value) >
            static_cast<std::uintmax_t>(
                std::numeric_limits<std::int64_t>::max())) {
        throw NotSupported(__FILE__, __LINE__, __func__,
                           "an integer above the int64 range");
    }
    return pnode{value};
}


template <typename ValueType>
inline std::enable_if_t<std::is_floating_point<ValueType>::value ||
                            std::is_same<ValueType, float16>::value ||
                            std::is_same<ValueType, bfloat16>::value,
                        pnode>
write_value(ValueType value)
{
    // float16 and bfloat16 only convert to float
    using float_type =
        std::conditional_t<std::is_floating_point<ValueType>::value, ValueType,
                           float>;
    return pnode{static_cast<double>(static_cast<float_type>(value))};
}


template <typename ValueType>
inline std::enable_if_t<gko::is_complex_s<ValueType>::value, pnode> write_value(
    const ValueType& value)
{
    return pnode{pnode::array_type{write_value(value.real()),
                                   write_value(value.imag())}};
}


// a case of a write_value switch that writes the enumerator by its name
#define GKO_WRITE_ENUM_CASE(_enum, _enumerator) \
    case _enum::_enumerator:                    \
        return pnode(#_enumerator)


inline pnode write_value(solver::initial_guess_mode mode)
{
    switch (mode) {
        GKO_WRITE_ENUM_CASE(solver::initial_guess_mode, zero);
        GKO_WRITE_ENUM_CASE(solver::initial_guess_mode, rhs);
        GKO_WRITE_ENUM_CASE(solver::initial_guess_mode, provided);
    }
    GKO_NOT_SUPPORTED(mode);
}


inline pnode write_value(precision_reduction value)
{
    return pnode{pnode::array_type{write_value(value.get_preserving()),
                                   write_value(value.get_nonpreserving())}};
}


inline pnode write_value(stop::mode mode)
{
    switch (mode) {
        GKO_WRITE_ENUM_CASE(stop::mode, absolute);
        GKO_WRITE_ENUM_CASE(stop::mode, initial_resnorm);
        GKO_WRITE_ENUM_CASE(stop::mode, rhs_norm);
    }
    GKO_NOT_SUPPORTED(mode);
}


/**
 * A configuration map with only the "type" key.
 */
inline pnode::map_type make_config_map(const std::string& type)
{
    return {{"type", pnode{type}}};
}


/**
 * A configuration map with the "type" and "value_type" keys.
 */
template <typename ValueType>
inline pnode::map_type make_config_map(const std::string& type)
{
    auto map = make_config_map(type);
    map["value_type"] = pnode{type_string<ValueType>::str()};
    return map;
}


/**
 * What write_config writes for a LinOp-valued parameter, which a configuration
 * can not contain.
 */
constexpr const char* object_placeholder = "<object>";


template <typename T>
inline void write_object(pnode::map_type& map, const std::string& key,
                         const std::shared_ptr<T>& object)
{
    if (object) {
        map[key] = pnode{object_placeholder};
    }
}


/**
 * Writes a nested factory under `key`, and leaves the key out if it is not
 * set: an empty node would be null in JSON or YAML, which the readers reject.
 */
inline void write_factory(pnode::map_type& map, const std::string& key,
                          const std::shared_ptr<const LinOpFactory>& factory)
{
    if (factory) {
        map[key] = write_config(*factory);
    }
}


bool is_identity_factory(const LinOpFactory& factory);


/**
 * write_factory for a parameter whose default is the identity (e.g. a
 * solver's `preconditioner`), which also leaves out a matrix::IdentityFactory.
 */
inline void write_factory_with_identity_default(
    pnode::map_type& map, const std::string& key,
    const std::shared_ptr<const LinOpFactory>& factory)
{
    if (factory && is_identity_factory(*factory)) {
        return;
    }
    write_factory(map, key, factory);
}


/**
 * Writes stopping criteria as an array, with combined criteria expanded.
 */
inline pnode write_criteria(
    const std::vector<std::shared_ptr<const stop::CriterionFactory>>& criteria)
{
    pnode::array_type result;
    for (const auto& criterion : criteria) {
        if (!criterion) {
            throw NotSupported(__FILE__, __LINE__, __func__,
                               "a null criterion");
        }
        auto written = write_config(*criterion);
        if (written.get_tag() == pnode::tag_t::array) {
            result.insert(result.end(), written.get_array().begin(),
                          written.get_array().end());
        } else {
            result.push_back(std::move(written));
        }
    }
    return pnode{result};
}


/**
 * Writes the parameters that common_solver_parse reads.
 */
template <typename SolverParam>
inline void write_common_solver(pnode::map_type& map, const SolverParam& params)
{
    write_object(map, "generated_preconditioner",
                 params.generated_preconditioner);
    map["criteria"] = write_criteria(params.criteria);
    write_factory_with_identity_default(map, "preconditioner",
                                        params.preconditioner);
    write_object(map, "nullspace", params.nullspace);
    write_object(map, "left_nullspace", params.left_nullspace);
}


/**
 * parameter_writer<T>::write writes the parameters of a T::Factory as the
 * configuration that T::parse reads. To add a class, declare its specialization
 * below with GKO_DECLARE_WRITER*, define write in the write_*.cpp of its group
 * and instantiate it there with GKO_INSTANTIATE_WRITER*, and add it to the
 * dispatch in write_config.cpp.
 */
template <typename T>
struct parameter_writer;


/**
 * Declares the parameter_writer of a class, of a class template over the value
 * type, or of one over the value and index type.
 */
#define GKO_DECLARE_WRITER(_class)                                 \
    template <>                                                    \
    struct parameter_writer<_class> {                              \
        static pnode write(const _class::parameters_type& params); \
    }

#define GKO_DECLARE_WRITER_VALUE_TYPE(_class)                           \
    template <typename ValueType>                                       \
    struct parameter_writer<_class<ValueType>> {                        \
        static pnode write(                                             \
            const typename _class<ValueType>::parameters_type& params); \
    }

#define GKO_DECLARE_WRITER_VALUE_AND_INDEX_TYPE(_class)                   \
    template <typename ValueType, typename IndexType>                     \
    struct parameter_writer<_class<ValueType, IndexType>> {               \
        static pnode write(                                               \
            const typename _class<ValueType, IndexType>::parameters_type& \
                params);                                                  \
    }


/**
 * Instantiates the parameter_writer of a class template for each value type,
 * or for each value and index type.
 */
#define GKO_WRITER_INSTANCE_(ValueType, _class) \
    struct parameter_writer<_class<ValueType>>
#define GKO_INSTANTIATE_WRITER_VALUE_TYPE(_class) \
    GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_VARGS(GKO_WRITER_INSTANCE_, _class)

#define GKO_WRITER_INDEX_INSTANCE_(ValueType, IndexType, _class) \
    struct parameter_writer<_class<ValueType, IndexType>>
#define GKO_INSTANTIATE_WRITER_VALUE_AND_INDEX_TYPE(_class)               \
    GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_VARGS(GKO_WRITER_INDEX_INSTANCE_, \
                                              int32, _class);             \
    GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_VARGS(GKO_WRITER_INDEX_INSTANCE_, \
                                              int64, _class)


GKO_DECLARE_WRITER(stop::Combined);
GKO_DECLARE_WRITER(stop::Iteration);
GKO_DECLARE_WRITER(stop::Time);
GKO_DECLARE_WRITER_VALUE_TYPE(stop::ResidualNorm);
GKO_DECLARE_WRITER_VALUE_TYPE(stop::ImplicitResidualNorm);
GKO_DECLARE_WRITER_VALUE_AND_INDEX_TYPE(preconditioner::Jacobi);
GKO_DECLARE_WRITER_VALUE_TYPE(solver::Cg);


}  // namespace config
}  // namespace gko


#endif  // GKO_CORE_CONFIG_WRITE_HELPER_HPP_
