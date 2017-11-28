#ifndef GKO_CORE_TYPES_HPP_
#define GKO_CORE_TYPES_HPP_


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


#define GINKGO_INSTANTIATE_FOR_EACH_VALUE_TYPE(_macro) \
    template _macro(float);                            \
    template _macro(double);                           \
    template _macro(std::complex<float>);              \
    template _macro(std::complex<double>)


}  // namespace gko


#endif  // GKO_CORE_TYPES_HPP_
