#ifndef GKO_CORE_TYPES_HPP_
#define GKO_CORE_TYPES_HPP_


#include <complex>
#include <cstddef>
#include <cstdint>


namespace gko {


/**
 * Integral type used for allocation quantities.
 */
using size_type = std::size_t;


/**
 * 32-bit signed integral type.
 */
using int32 = std::int32_t;


/**
 * 64-bit signed integral type.
 */
using int64 = std::int64_t;


/**
 * The most precise floating-point type.
 */
using full_precision = double;


/**
 * Precision used if no precision is explicitly specified.
 */
using default_precision = double;


/**
 * Calls a given macro for each executor type.
 *
 * The macro should take two parameters:
 *
 * -   the first one is replaced with the executor class name
 * -   the second one with the executor short name (used for namespace name)
 *
 * @param _enable_macro  macro name which will be called
 *
 * @note  the macro is not called for ReferenceExecutor
 */
#define GKO_ENABLE_FOR_ALL_EXECUTORS(_enable_macro) \
    _enable_macro(CpuExecutor, cpu);                \
    _enable_macro(GpuExecutor, gpu)


/**
 * Instantiates a template for each value type compiled by Ginkgo.
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take one argument, which is replaced by the
 *                value type.
 */
#define GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(_macro) \
    template _macro(float);                         \
    template _macro(double);                        \
    template _macro(std::complex<float>);           \
    template _macro(std::complex<double>)


}  // namespace gko


#endif  // GKO_CORE_TYPES_HPP_
