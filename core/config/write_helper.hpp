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
#include <ginkgo/core/factorization/ic.hpp>
#include <ginkgo/core/factorization/ilu.hpp>
#include <ginkgo/core/factorization/incomplete_factorization.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/preconditioner/ic.hpp>
#include <ginkgo/core/preconditioner/ilu.hpp>
#include <ginkgo/core/preconditioner/isai.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/solver/triangular.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/criterion.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/core/stop/time.hpp>

#include "core/config/type_descriptor_helper.hpp"


namespace gko {
namespace config {


/**
 * write_value writes a parameter value in the encoding that get_value (see
 * core/config/config_helper.hpp) reads it from. There is one overload for
 * each get_value overload.
 *
 * This is the overload for bool.
 */
inline pnode write_value(bool value) { return pnode{value}; }


/**
 * write_value for integral types (except bool)
 *
 * @throw NotSupported  if the value is above the int64 range, which get_value
 *                      can not read
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


/**
 * write_value for floating point types, written as a real (double) value.
 * get_value converts it back without loss.
 */
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


/**
 * write_value for complex types, written as the array [real, imag]
 */
template <typename ValueType>
inline std::enable_if_t<gko::is_complex_s<ValueType>::value, pnode> write_value(
    const ValueType& value)
{
    return pnode{pnode::array_type{write_value(value.real()),
                                   write_value(value.imag())}};
}


/**
 * write_value for initial_guess_mode, written as the strings get_value reads
 */
inline pnode write_value(solver::initial_guess_mode mode)
{
    switch (mode) {
    case solver::initial_guess_mode::zero:
        return pnode{"zero"};
    case solver::initial_guess_mode::rhs:
        return pnode{"rhs"};
    case solver::initial_guess_mode::provided:
        return pnode{"provided"};
    }
    GKO_NOT_SUPPORTED(mode);
}


/**
 * write_value for the orthogonalization method of Gmres, written as the
 * strings Gmres::parse reads
 */
inline pnode write_value(solver::gmres::ortho_method method)
{
    switch (method) {
    case solver::gmres::ortho_method::mgs:
        return pnode{"mgs"};
    case solver::gmres::ortho_method::cgs:
        return pnode{"cgs"};
    case solver::gmres::ortho_method::cgs2:
        return pnode{"cgs2"};
    }
    GKO_NOT_SUPPORTED(method);
}


/**
 * write_value for the algorithm of the triangular solvers, written as the
 * strings common_trisolver_parse reads
 */
inline pnode write_value(solver::trisolve_algorithm algorithm)
{
    switch (algorithm) {
    case solver::trisolve_algorithm::sparselib:
        return pnode{"sparselib"};
    case solver::trisolve_algorithm::syncfree:
        return pnode{"syncfree"};
    }
    GKO_NOT_SUPPORTED(algorithm);
}


/**
 * write_value for the SpMV strategy of the factors of the factorizations,
 * written as the strings get_strategy (see core/config/config_helper.hpp)
 * reads
 */
inline pnode write_value(matrix::csr::spmv_strategy strategy)
{
    switch (strategy) {
    case matrix::csr::spmv_strategy::automatic:
        return pnode{"automatic"};
    case matrix::csr::spmv_strategy::load_balance:
        return pnode{"load_balance"};
    case matrix::csr::spmv_strategy::merge_path:
        return pnode{"merge_path"};
    case matrix::csr::spmv_strategy::classical:
        return pnode{"classical"};
    case matrix::csr::spmv_strategy::sparselib:
        return pnode{"sparselib"};
    }
    GKO_NOT_SUPPORTED(strategy);
}


/**
 * write_value for the algorithm of the incomplete factorizations, written as
 * the strings factorization::Ilu::parse and factorization::Ic::parse read
 */
inline pnode write_value(factorization::incomplete_algorithm algorithm)
{
    switch (algorithm) {
    case factorization::incomplete_algorithm::sparselib:
        return pnode{"sparselib"};
    case factorization::incomplete_algorithm::syncfree:
        return pnode{"syncfree"};
    }
    GKO_NOT_SUPPORTED(algorithm);
}


/**
 * write_value for the kind of ISAI, written as the strings the ISAI parser
 * (core/config/preconditioner_isai_config.cpp) selects the class with
 */
inline pnode write_value(preconditioner::isai_type type)
{
    switch (type) {
    case preconditioner::isai_type::lower:
        return pnode{"lower"};
    case preconditioner::isai_type::upper:
        return pnode{"upper"};
    case preconditioner::isai_type::general:
        return pnode{"general"};
    case preconditioner::isai_type::spd:
        return pnode{"spd"};
    }
    GKO_NOT_SUPPORTED(type);
}


/**
 * write_value for precision_reduction, written as the array
 * [preserving, nonpreserving]
 */
inline pnode write_value(precision_reduction value)
{
    return pnode{pnode::array_type{write_value(value.get_preserving()),
                                   write_value(value.get_nonpreserving())}};
}


/**
 * write_value for the baseline of the residual norm criteria, written as the
 * strings get_mode (core/config/stop_config.cpp) reads
 */
inline pnode write_value(stop::mode mode)
{
    switch (mode) {
    case stop::mode::absolute:
        return pnode{"absolute"};
    case stop::mode::initial_resnorm:
        return pnode{"initial_resnorm"};
    case stop::mode::rhs_norm:
        return pnode{"rhs_norm"};
    }
    GKO_NOT_SUPPORTED(mode);
}


/**
 * Starts the configuration of a class without template parameters: a map with
 * the key `"type"`, set to the key that selects the class in parse (the
 * registry key, or the criterion name).
 */
inline pnode::map_type make_config_map(const std::string& type)
{
    return {{"type", pnode{type}}};
}


/**
 * Starts the configuration of a class with a value type: a map with the keys
 * `"type"` and `"value_type"`.
 */
template <typename ValueType>
inline pnode::map_type make_config_map(const std::string& type)
{
    auto map = make_config_map(type);
    map["value_type"] = pnode{type_string<ValueType>::str()};
    return map;
}


/**
 * The placeholder that write_config writes for a LinOp-valued parameter,
 * since a configuration can not contain the object itself. parse reads it as
 * the name of an object in the registry.
 */
constexpr const char* object_placeholder = "<object>";


/**
 * Writes a LinOp-valued parameter (e.g. `generated_preconditioner`) as the
 * object_placeholder if it is set, and leaves it out if it is not.
 */
template <typename T>
inline void write_object(pnode::map_type& map, const std::string& key,
                         const std::shared_ptr<T>& object)
{
    if (object) {
        map[key] = pnode{object_placeholder};
    }
}


/**
 * Writes a nested factory parameter as the factory's configuration under
 * `key`, and leaves the key out if the factory is not set. parse leaves a
 * parameter whose key is missing at its default. An empty node instead would
 * be `null` in a JSON or YAML record, which their readers reject.
 */
inline void write_factory(pnode::map_type& map, const std::string& key,
                          const std::shared_ptr<const LinOpFactory>& factory)
{
    if (factory) {
        map[key] = write_config(*factory);
    }
}


/**
 * Returns whether `factory` is a matrix::IdentityFactory, of any value type.
 * It has no writer of its own.
 */
bool is_identity_factory(const LinOpFactory& factory);


/**
 * write_factory for a parameter whose default, when it is not set, acts as the
 * identity, e.g. a solver's `preconditioner`. A matrix::IdentityFactory is
 * then left out too, since parse's default gives an equivalent object.
 *
 * Only use it where the default is the identity. Elsewhere (e.g. ILU's
 * `l_solver`, whose default is a triangular solver), use write_factory, which
 * throws NotSupported for an IdentityFactory.
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
 * Writes stopping criteria as an array of criterion maps. A combined criterion
 * (see stop::combine) is expanded into the criteria it combines.
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
 * Writes the parameters that common_solver_parse (see
 * core/config/solver_config.hpp) reads.
 */
template <typename SolverParam>
inline void write_common_solver(pnode::map_type& map, const SolverParam& params)
{
    write_object(map, "generated_preconditioner",
                 params.generated_preconditioner);
    map["criteria"] = write_criteria(params.criteria);
    // an unset preconditioner is the identity (see
    // EnablePreconditionedIterativeSolver::generate_preconditioner)
    write_factory_with_identity_default(map, "preconditioner",
                                        params.preconditioner);
    write_object(map, "nullspace", params.nullspace);
    write_object(map, "left_nullspace", params.left_nullspace);
}


/**
 * Writes the parameters that common_trisolver_parse (see
 * core/config/trisolver_config.hpp) reads.
 */
template <typename SolverParam>
inline void write_common_trisolver(pnode::map_type& map,
                                   const SolverParam& params)
{
    map["num_rhs"] = write_value(params.num_rhs);
    map["unit_diagonal"] = write_value(params.unit_diagonal);
    map["algorithm"] = write_value(params.algorithm);
}


/**
 * parameter_writer<T>::write writes the parameters of the factory of the
 * concrete class `T` (e.g. solver::Cg<double>) as the configuration that parse
 * reads for `T`: the keys `T::parse` reads, plus `"type"` and `"value_type"`.
 *
 * To add a writer for a class (template):
 * 1. specialize parameter_writer for it below;
 * 2. define `write` in the write_*.cpp file of its group, and explicitly
 *    instantiate it for the same types that parse dispatches over (e.g.
 *    GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE where the parser uses
 *    GKO_PARSE_VALUE_TYPE);
 * 3. add it, with the same type lists, to the dispatch in write_config.cpp.
 *    A class template with a non-type template parameter (e.g. the
 *    ReverseApply of preconditioner::Ilu) gets one alias template for each
 *    value that parse selects, and one dispatch line for each alias.
 *
 * @tparam T  the concrete class
 */
template <typename T>
struct parameter_writer;


template <>
struct parameter_writer<stop::Combined> {
    /** @return the array of the combined criteria, see write_criteria */
    static pnode write(const stop::Combined::parameters_type& params);
};


template <>
struct parameter_writer<stop::Iteration> {
    static pnode write(const stop::Iteration::parameters_type& params);
};


template <>
struct parameter_writer<stop::Time> {
    static pnode write(const stop::Time::parameters_type& params);
};


template <typename ValueType>
struct parameter_writer<stop::ResidualNorm<ValueType>> {
    static pnode write(
        const typename stop::ResidualNorm<ValueType>::parameters_type& params);
};


template <typename ValueType>
struct parameter_writer<stop::ImplicitResidualNorm<ValueType>> {
    static pnode write(
        const typename stop::ImplicitResidualNorm<ValueType>::parameters_type&
            params);
};


template <typename ValueType, typename IndexType>
struct parameter_writer<preconditioner::Jacobi<ValueType, IndexType>> {
    static pnode write(const typename preconditioner::Jacobi<
                       ValueType, IndexType>::parameters_type& params);
};


template <typename ValueType>
struct parameter_writer<solver::Cg<ValueType>> {
    static pnode write(
        const typename solver::Cg<ValueType>::parameters_type& params);
};


template <typename ValueType>
struct parameter_writer<solver::Bicgstab<ValueType>> {
    static pnode write(
        const typename solver::Bicgstab<ValueType>::parameters_type& params);
};


template <typename ValueType>
struct parameter_writer<solver::Gmres<ValueType>> {
    static pnode write(
        const typename solver::Gmres<ValueType>::parameters_type& params);
};


template <typename ValueType>
struct parameter_writer<solver::Ir<ValueType>> {
    static pnode write(
        const typename solver::Ir<ValueType>::parameters_type& params);
};


template <typename ValueType, typename IndexType>
struct parameter_writer<solver::LowerTrs<ValueType, IndexType>> {
    static pnode write(
        const typename solver::LowerTrs<ValueType, IndexType>::parameters_type&
            params);
};


template <typename ValueType, typename IndexType>
struct parameter_writer<solver::UpperTrs<ValueType, IndexType>> {
    static pnode write(
        const typename solver::UpperTrs<ValueType, IndexType>::parameters_type&
            params);
};


template <typename ValueType, typename IndexType>
struct parameter_writer<factorization::Ilu<ValueType, IndexType>> {
    static pnode write(const typename factorization::Ilu<
                       ValueType, IndexType>::parameters_type& params);
};


template <typename ValueType, typename IndexType>
struct parameter_writer<factorization::Ic<ValueType, IndexType>> {
    static pnode write(
        const typename factorization::Ic<ValueType, IndexType>::parameters_type&
            params);
};


/** writes `"reverse_apply"`, with which parse selects the instantiation */
template <typename ValueType, bool ReverseApply, typename IndexType>
struct parameter_writer<
    preconditioner::Ilu<ValueType, ReverseApply, IndexType>> {
    static pnode write(
        const typename preconditioner::Ilu<ValueType, ReverseApply,
                                           IndexType>::parameters_type& params);
};


template <typename ValueType, typename IndexType>
struct parameter_writer<preconditioner::Ic<ValueType, IndexType>> {
    static pnode write(const typename preconditioner::Ic<
                       ValueType, IndexType>::parameters_type& params);
};


/** writes `"isai_type"`, with which parse selects the instantiation */
template <preconditioner::isai_type IsaiType, typename ValueType,
          typename IndexType>
struct parameter_writer<preconditioner::Isai<IsaiType, ValueType, IndexType>> {
    static pnode write(
        const typename preconditioner::Isai<
            IsaiType, ValueType, IndexType>::parameters_type& params);
};


}  // namespace config
}  // namespace gko


#endif  // GKO_CORE_CONFIG_WRITE_HELPER_HPP_
